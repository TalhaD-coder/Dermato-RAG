"""Pipeline baslangic testi - LLM cagrisi yapmaz, sadece yukleme kontrol eder."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

print("1. Import kontrol ediliyor...")
from src.models.vision_encoder import DermatoVisionEncoder
print("   vision_encoder OK")

from src.llm.generator import DiagnosticGenerator
print("   generator OK")

from src.llm.confidence import ConfidenceEvaluator
print("   confidence OK")

print("\n2. Model dosyasi kontrol ediliyor...")
model_path = Path("models/best_model.pt")
print(f"   best_model.pt: {'MEVCUT (' + str(round(model_path.stat().st_size/1e6)) + ' MB)' if model_path.exists() else 'EKSIK!'}")

print("\n3. ChromaDB kontrol ediliyor...")
import chromadb
client = chromadb.PersistentClient(path="data/embeddings/chromadb")
col = client.get_or_create_collection("dermato_kb")
print(f"   Koleksiyon: dermato_kb | Dokuman sayisi: {col.count()}")

print("\n4. Vision model yukleniyor (bu biraz surabilir)...")
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"   Device: {device}")
model = DermatoVisionEncoder.load_checkpoint("models/best_model.pt", device=device)
model.eval()
print("   Vision model yuklendi!")

print("\n--- TUM KONTROLLER GECTI ---")
