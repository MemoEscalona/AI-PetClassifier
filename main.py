import argparse
import os
from app import create_app
from app.train import entrenar_modelo
from app.config import RUTA_MODELO

def main():
    parser = argparse.ArgumentParser(description="Clasificador de Perros y Gatos 🐶🐱")
    parser.add_argument(
        "--train", action="store_true", help="Entrenar el modelo desde cero"
    )
    parser.add_argument(
        "--serve", action="store_true", help="Ejecutar el servidor Flask"
    )
    args = parser.parse_args()

    if args.train:
        print("🔄 Iniciando entrenamiento del modelo...")
        entrenar_modelo()
        print(f"✅ Modelo entrenado y guardado en {RUTA_MODELO}")

    elif args.serve:
        print("🚀 Iniciando servidor Flask en http://localhost:5000")
        app = create_app()
        app.run(host="0.0.0.0", port=5000, debug=True)

    else:
        print("❌ Debes especificar una opción: --train o --serve")

if __name__ == "__main__":
    main()
