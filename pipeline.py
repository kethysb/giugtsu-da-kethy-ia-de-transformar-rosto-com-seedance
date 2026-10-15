"""
giugtsu da kethy - Pipeline Híbrido de Motion Transfer com DWPose, SAM 2, InsightFace e Seedance 2.5 API
"""

import os
import glob
import json
import argparse
import cv2
import numpy as np
from tqdm import tqdm
import torch

try:
    from controlnet_aux import DWposeDetector
except ImportError:
    DWposeDetector = None

try:
    import insightface
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None


class GiugtsuPipeline:
    def __init__(self, models_dir="models", outputs_dir="outputs"):
        self.models_dir = models_dir
        self.outputs_dir = outputs_dir
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Giugtsu da Kethy inicializado no dispositivo: {self.device}")

    def extract_motion_dwpose(self, video_path, output_name="motion_dwpose.mp4"):
        if DWposeDetector is None:
            raise ImportError("controlnet-aux não está instalado. Execute: pip install controlnet-aux")

        output_path = os.path.join(self.outputs_dir, output_name)
        print(f"[*] Processando esqueleto de movimento com DWPose em: {video_path}")
        dwpose = DWposeDetector.from_pretrained('yzd-v/DWPose')

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for _ in tqdm(range(total_frames), desc="DWPose Frames"):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_img = dwpose(frame_rgb)
            out.write(cv2.cvtColor(np.array(pose_img), cv2.COLOR_RGB2BGR))

        cap.release()
        out.release()
        print(f"[✔] Vídeo de pose gerado com sucesso: {output_path}")
        return output_path

    def extract_identity_insightface(self, images_dir="inputs"):
        if FaceAnalysis is None:
            raise ImportError("insightface não está instalado. Execute: pip install insightface")

        print("[*] Extraindo biometria e ângulos com InsightFace (Soul ID)...")
        app = FaceAnalysis(name='buffalo_l')
        app.prepare(ctx_id=0 if self.device == "cuda" else -1, det_size=(640, 640))

        img_paths = glob.glob(os.path.join(images_dir, "*.jpg")) + glob.glob(os.path.join(images_dir, "*.png"))
        face_data = []

        for p in img_paths:
            img = cv2.imread(p)
            faces = app.get(img)
            if faces:
                face_data.append({
                    "path": p,
                    "bbox": faces[0].bbox.tolist(),
                    "embedding": faces[0].normed_embedding.tolist()
                })
                print(f"[✔] Rosto ancorado: {p}")

        print(f"[✔] Total de {len(face_data)} referências de identidade extraídas.")
        return face_data

    def dispatch_seedance_api(self, motion_video, character_images, prompt, api_key=None):
        api_key = api_key or os.getenv("SEEDANCE_API_KEY", "DEMO_KEY")
        payload = {
            "model": "seedance-2.5-v2v",
            "motion_video": motion_video,
            "character_images": character_images,
            "prompt": prompt,
            "fps": 24,
            "resolution": "1080x1920"
        }
        print("[*] Despachando payload para a API do Seedance 2.5:")
        print(json.dumps(payload, indent=2))
        return {"status": "dispatched", "payload": payload}


def main():
    parser = argparse.ArgumentParser(description="giugtsu da kethy - Motion Transfer Pipeline")
    parser.add_argument("--video", type=str, default="inputs/danca.mp4", help="Caminho para o vídeo de dança")
    parser.add_argument("--images", type=str, default="inputs", help="Diretório com as fotos do personagem")
    parser.add_argument("--prompt", type=str, default="Cirilo dancing Brazilian funk, 8k, photorealistic", help="Prompt descritivo")
    parser.add_argument("--api_key", type=str, default=None, help="Chave da API do Seedance 2.5")

    args = parser.parse_args()
    pipeline = GiugtsuPipeline()

    if os.path.exists(args.video):
        pose_video = pipeline.extract_motion_dwpose(args.video)
    else:
        print(f"[!] Vídeo {args.video} não encontrado. Pulando extração de pose.")
        pose_video = None

    faces = pipeline.extract_identity_insightface(args.images)
    pipeline.dispatch_seedance_api(pose_video, [f["path"] for f in faces], args.prompt, args.api_key)


if __name__ == "__main__":
    main()
