import wikipediaapi
import json

# Lista de videojuegos
GAMES = [
    "The Legend of Zelda: Breath of the Wild",
    "The Witcher 3: Wild Hunt",
    "Red Dead Redemption 2",
    "Grand Theft Auto V",
    "Minecraft",
    "Dark Souls",
    "The Last of Us",
    "God of War (2018 video game)",
    "Elden Ring",
    "Super Mario Odyssey",
    "Halo: Combat Evolved",
    "Portal (video game)",
    "Half-Life 2",
    "Doom (1993 video game)",
    "Cyberpunk 2077",
    "Hollow Knight",
    "Stardew Valley",
    "Among Us",
    "Undertale",
    "Celeste (video game)",
]

def fetch_corpus():
    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="SearchEngineProject/1.0"
    )

    corpus = []

    for title in GAMES:
        print(f"Fetching: {title}...")
        page = wiki.page(title)

        if not page.exists():
            print(f"  [WARNING] Page not found: {title}")
            continue

        # Tomar solo los primeros 1000 caracteres para mantenerlo manejable
        text = page.summary
        word_count = len(text.split())

        if word_count < 50:
            print(f"  [WARNING] Too short ({word_count} words), skipping.")
            continue

        doc = {
            "id": len(corpus) + 1,
            "title": page.title,
            "text": text,
            "source": page.fullurl,
            "word_count": word_count
        }
        corpus.append(doc)
        print(f"  OK — {word_count} words")

    # Guardar corpus.json
    with open("corpus.json", "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"\nDone! {len(corpus)} documents saved to corpus.json")

if __name__ == "__main__":
    fetch_corpus()
