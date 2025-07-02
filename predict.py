import os, shutil, time, uuid, subprocess, sys, requests, json
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        # ------------------------------------------------------------------
        # Make sure the ComfyUI source tree exists in the container.
        # ------------------------------------------------------------------
        if not os.path.isdir("ComfyUI"):
            subprocess.check_call(
                [
                    "git", "clone", "--depth", "1",
                    "https://github.com/comfyanonymous/ComfyUI.git"
                ]
            )
        sys.path.append(os.path.abspath("ComfyUI"))

        # ------------------------------------------------------------------
        # Create runtime directories
        # ------------------------------------------------------------------
        os.makedirs("ComfyUI/models/checkpoints", exist_ok=True)
        os.makedirs("ComfyUI/models/controlnet",   exist_ok=True)
        os.makedirs("ComfyUI/input",               exist_ok=True)
        os.makedirs("ComfyUI/output",              exist_ok=True)

        # ------------------------------------------------------------------
        # Download model weights from Hugging Face if not cached
        # ------------------------------------------------------------------
        ckpt_path = (
            "ComfyUI/models/checkpoints/"
            "SaliaHighlady.safetensors"
        )
        cn_path = (
            "ComfyUI/models/controlnet/"
            "diffusion_pytorch_model_promax.safetensors"
        )

        if not os.path.exists(ckpt_path):
            print("Downloading base model weights …")
            url = (
                "https://huggingface.co/"
                "saliacoel/comfy_weights/resolve/main/"
                "SaliaHighlady.safetensors"
            )
            self._stream_download(url, ckpt_path)

        if not os.path.exists(cn_path):
            print("Downloading ControlNet weights …")
            url = (
                "https://huggingface.co/"
                "saliacoel/comfy_weights/resolve/main/"
                "diffusion_pytorch_model_promax.safetensors"
            )
            self._stream_download(url, cn_path)

        # ------------------------------------------------------------------
        # Launch ComfyUI headless server
        # ------------------------------------------------------------------
        print("Starting ComfyUI server …")
        self.server_process = subprocess.Popen(
            ["python", "-m", "ComfyUI"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Wait until the server responds
        for _ in range(30):
            try:
                requests.get("http://127.0.0.1:8188/prompt")
                print("ComfyUI server is ready.")
                break
            except Exception:
                time.sleep(1)
        else:
            raise RuntimeError("ComfyUI server failed to start.")

    # ----------------------------------------------------------------------
    # Private helper to download large files with streamed chunks
    # ----------------------------------------------------------------------
    @staticmethod
    def _stream_download(url, dest, chunk=8192):
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for block in r.iter_content(chunk_size=chunk):
                if block:
                    f.write(block)

    # ----------------------------------------------------------------------
    # Prediction endpoint
    # ----------------------------------------------------------------------
    def predict(
        self,
        image: Path = Input(description="Input PNG image"),
        prompt: str  = Input(description="Positive text prompt"),
        negative_prompt: str = Input(description="Negative text prompt")
    ) -> Path:

        # Load base workflow JSON
        with open("workflow_api.json", "r") as f:
            workflow = json.load(f)

        # Replace prompt text
        id_to_class = {
            node_id: node_data["class_type"]
            for node_id, node_data in workflow.items()
        }
        sampler_node = next(
            node for node, cls in id_to_class.items() if "KSampler" in cls
        )
        pos_node = workflow[sampler_node]["inputs"]["positive"][0]
        neg_node = workflow[sampler_node]["inputs"]["negative"][0]
        workflow[pos_node]["inputs"]["text"] = prompt
        workflow[neg_node]["inputs"]["text"] = negative_prompt

        # Handle image input
        if any(cls in ("LoadImage", "ImageLoader")
               for cls in id_to_class.values()):
            load_node = next(
                node for node, cls in id_to_class.items()
                if cls in ("LoadImage", "ImageLoader")
            )
            shutil.copy(str(image), "ComfyUI/input/input.png")
            workflow[load_node]["inputs"]["image"] = "input.png"

        # Send workflow to ComfyUI API
        client_id = str(uuid.uuid4())
        resp = requests.post(
            "http://127.0.0.1:8188/prompt",
            json={"prompt": workflow, "client_id": client_id},
        )
        resp.raise_for_status()
        prompt_id = resp.json().get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI queue error: {resp.json()}")

        # Wait for new output file
        previous = {
            os.path.join(root, f)
            for root, _, files in os.walk("ComfyUI/output")
            for f in files
        }
        output_path = None
        for _ in range(120):
            time.sleep(1)
            for root, _, files in os.walk("ComfyUI/output"):
                for f in files:
                    p = os.path.join(root, f)
                    if p not in previous and p.endswith((".png", ".jpg")):
                        output_path = p
                        break
                if output_path:
                    break
            if output_path:
                break
        if output_path is None:
            raise RuntimeError("No image produced within timeout.")

        return Path(output_path)
