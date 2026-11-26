import math
import joblib
import pandas
from flask import Flask, render_template, request

app = Flask(__name__)

modele = joblib.load("monModelML.joblib")


@app.route("/", methods=["GET", "POST"])
def index():
  resultat = None
  erreur = None
  heures = request.form.get("heures_etude", "")
  note = request.form.get("note_partiel", "")

  if request.method == "POST":
    try:
      h, n = float(heures), float(note)
      if not (
          math.isfinite(h) and math.isfinite(n) and 0 <= h <= 10 and 0 <= n <= 10
      ):
        raise ValueError()
        
      X = pandas.DataFrame([[h, n]], columns=["heures_etude", "note_partiel"])
      prediction = int(modele.predict(X)[0])
      resultat = {
          0: "Vous avez une forte probabilité d'échouer",
          1: "Vous avez une forte probabilité de réussir",
      }[prediction]

    except ValueError:
      erreur = "Saisir les heures et notes comprises entre 0 et 10"

  return render_template(
      "index.html", resultat=resultat, erreur=erreur, heures=heures, note=note
  )


if __name__ == "__main__":
  app.run(debug=True)