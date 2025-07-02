import os, shutil, time, uuid, subprocess, requests, json
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load models and start ComfyUI server (runs once when the container starts)."""
        # Create directories for ComfyUI and model weights
        os.makedirs("ComfyUI/models/checkpoints", exist_ok=True)
        os.makedirs("ComfyUI/models/controlnet", exist_ok=True)
        os.makedirs("ComfyUI/input", exist_ok=True)
        os.makedirs("ComfyUI/output", exist_ok=True)
        
        # Download model weights from Hugging Face if not already cached
        ckpt_path = "ComfyUI/models/checkpoints/SaliaHighlady.safetensors"
        cn_path   = "ComfyUI/models/controlnet/diffusion_pytorch_model_promax.safetensors"
        if not os.path.exists(ckpt_path):
            print("Downloading base model weights...")
            url = "https://huggingface.co/saliacoel/comfy_weights/resolve/main/SaliaHighlady.safetensors"
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(ckpt_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        if not os.path.exists(cn_path):
            print("Downloading ControlNet model weights...")
            url = "https://huggingface.co/saliacoel/comfy_weights/resolve/main/diffusion_pytorch_model_promax.safetensors"
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(cn_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        # Start ComfyUI server in a background process (listening on default port 8188)
        print("Starting ComfyUI server...")
        # Use `-m ComfyUI` if ComfyUI is installed as a package. 
        # The server runs with a web API at 127.0.0.1:8188 by default.
        self.server_process = subprocess.Popen(
            ["python", "-m", "ComfyUI"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        # Wait for the server to be up by polling the /prompt endpoint
        for _ in range(30):
            try:
                requests.get("http://127.0.0.1:8188/prompt")
                print("ComfyUI server is ready.")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("ComfyUI server failed to start.")
    
    def predict(
        self,
        image: Path = Input(description="Input PNG image"),
        prompt: str = Input(description="Positive text prompt"),
        negative_prompt: str = Input(description="Negative text prompt")
    ) -> Path:
        """Run the ComfyUI workflow on the input image and prompts, return a 1024x1024 PNG."""
        # Load the ComfyUI API workflow JSON from file
        with open("workflow_api.json", "r") as f:
            workflow = json.load(f)
        
        # Update the workflow with the user-provided prompts
        # Find the KSampler (sampler) node and the associated text nodes for prompts
        id_to_class = {node_id: node_data["class_type"] for node_id, node_data in workflow.items()}
        # Identify the sampler node (usually class_type "KSampler" or similar)
        sampler_node = next(node for node, cls in id_to_class.items() if "KSampler" in cls)
        # The sampler's inputs will reference positive and negative text-encode nodes
        positive_text_node = workflow[sampler_node]["inputs"]["positive"][0]
        negative_text_node = workflow[sampler_node]["inputs"]["negative"][0]
        # Set the new prompt texts
        workflow[positive_text_node]["inputs"]["text"] = prompt
        workflow[negative_text_node]["inputs"]["text"] = negative_prompt
        
        # If there's an image input node (e.g. a LoadImage or ControlNet loader), update its path
        load_img_nodes = [node for node, cls in id_to_class.items() if cls in ("LoadImage", "ImageLoader")]
        if load_img_nodes:
            image_node = load_img_nodes[0]
            # Copy the user input image to ComfyUI/input as "input.png"
            shutil.copy(str(image), "ComfyUI/input/input.png")
            # Set the image node to use the filename of the uploaded image
            workflow[image_node]["inputs"]["image"] = "input.png"
        
        # Submit the workflow to the ComfyUI server via its HTTP API
        client_id = str(uuid.uuid4())  # unique client ID for this request
        payload = {"prompt": workflow, "client_id": client_id}
        resp = requests.post("http://127.0.0.1:8188/prompt", json=payload)
        resp.raise_for_status()
        result = resp.json()
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI failed to queue prompt: {result}")
        
        # Wait for the workflow execution to finish by polling the output directory for new file
        prev_files = set()
        for root, _, files in os.walk("ComfyUI/output"):
            for fname in files:
                prev_files.add(os.path.join(root, fname))
        # Poll for new output file(s)
        output_file_path = None
        for _ in range(120):  # up to 120 seconds
            time.sleep(1)
            new_file_found = False
            for root, _, files in os.walk("ComfyUI/output"):
                for fname in files:
                    fpath = os.path.join(root, fname)
                    if fpath.endswith((".png", ".jpg", ".jpeg")) and fpath not in prev_files:
                        output_file_path = fpath
                        new_file_found = True
                        break
                if new_file_found:
                    break
            if new_file_found:
                break
        if output_file_path is None:
            raise RuntimeError("No output image was produced by the workflow within the timeout.")
        
        # Return the path to the output image (1024x1024 PNG as specified by the workflow)
        return Path(output_file_path)
