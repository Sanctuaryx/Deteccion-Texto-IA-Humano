import json
import pathlib
import random

random.seed(42)


def gen_sample(i: int, ia: bool = False) -> dict:
    if ia:
        text = (
            "Como modelo de lenguaje, proporcionaré una respuesta neutral y bien "
            "estructurada sobre el tema. En conclusión, los factores expuestos permiten "
            "sintetizar la información."
        )
    else:
        text = (
            "Este es un texto humano con opiniones personales, giros coloquiales y "
            "posibles errores. No siempre es perfectamente estructurado."
        )
    return {
        "id": i,
        "prompt_id": i // 2,
        "text": f"{text} [{i}]",
        "label": 1 if ia else 0,  # 1 = IA, 0 = humano
    }


def main() -> None:
    data_dir = pathlib.Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    train = [gen_sample(i, ia=(i % 3 == 0)) for i in range(200)]
    val = [gen_sample(1000 + i, ia=(i % 4 == 0)) for i in range(60)]

    with open(data_dir / "train.jsonl", "w", encoding="utf-8") as f:
        for row in train:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    with open(data_dir / "val.jsonl", "w", encoding="utf-8") as f:
        for row in val:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Dummy dataset creado en data/train.jsonl y data/val.jsonl")


if __name__ == "__main__":
    main()
