import os

# Chemin vers ton dossier d’images
dossier = r"C:\Users\user\Desktop\projetCorpsEtrangers\data"

# Liste des fichiers triés
fichiers = sorted(os.listdir(dossier))

# Compteur
i = 1

for fichier in fichiers:
    if fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
        
        ancien_chemin = os.path.join(dossier, fichier)
        
        # garder l'extension originale
        ext = os.path.splitext(fichier)[1]
        nouveau_nom = f"poulet_{i}{ext}"
        
        nouveau_chemin = os.path.join(dossier, nouveau_nom)
        
        os.rename(ancien_chemin, nouveau_chemin)
        i += 1

print("Renommage terminé ! ✅")