import time  # Importer le module time pour mesurer la durée
from nim import train, play, plot_eval

# Mesurer le temps avant l'entraînement
start_time = time.time()

# Entraîner l'IA
ai, eval = train(10000)

# Mesurer le temps après l'entraînement
end_time = time.time()

# Calculer la durée totale de l'entraînement
training_time = end_time - start_time
print(f"Temps d'entraînement : {training_time:.2f} secondes")

# Afficher les résultats de l'évaluation
plot_eval(eval)

# Jouer contre l'IA (optionnel)
play(ai)
