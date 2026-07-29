import json

def tri_nom(mots_voulus,json_dir):
    # 1. Charger le fichier JSON
    with open(json_dir, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 2. Liste des mots voulus
    #mots_voulus = ["before", "thin", "cool", "go", "computer", "cousin", "help", "hello", "where"]

    # 3. Vérifier que chaque mot existe bien dans le dataset
    glosses_disponibles = {entry["gloss"] for entry in data}
    mots_manquants = [mot for mot in mots_voulus if mot not in glosses_disponibles]
    if mots_manquants:
        print(f" Attention, ces mots n'existent pas dans le dataset : {mots_manquants}")

    # 4. Filtrer les entrées qui correspondent aux mots voulus, dans l'ordre de mots_voulus
    data_par_gloss = {entry["gloss"]: entry for entry in data}
    data_filtree = [data_par_gloss[mot] for mot in mots_voulus if mot in data_par_gloss]

    print("Mots retenus :")
    for entry in data_filtree:
        print(f"{entry['gloss']:<12}-> {len(entry['instances'])} vidéos")
    json_filtre = json.dumps(data_filtree,indent=4,ensure_ascii=False)
    print(f"\n Fichier JSON est filtré  avec succès")
    return json_filtre