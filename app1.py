from flask import Flask, render_template, request
import random
import pickle
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms

# Flask app initialization
app = Flask(__name__)

# Load pre-trained models
with open('random_forest.pkl', 'rb') as file:
    rf_classifier = pickle.load(file)

with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

data = pd.read_csv('soil.impact.csv')
impact_scores = {'restorative': 1, 'neutral': 0, 'depleting': -1}
data['Impact_Score'] = data['Impact'].map(impact_scores)
impact_data = data[['Name', 'Impact', 'Soil_Type']].drop_duplicates()

# DEAP Genetic Algorithm setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()

# Helper function to filter crops by soil type
def filter_crops_by_soil(soil_type):
    return impact_data[impact_data['Soil_Type'] == soil_type]

# Genetic Algorithm: Create individual
def create_individual(soil_type, num_periods, periods_per_year):
    filtered_crops = filter_crops_by_soil(soil_type).values.tolist()
    return creator.Individual([random.choice(filtered_crops) for _ in range(num_periods)])

# Genetic Algorithm: Evaluate function
def evaluate(individual, periods_per_year):
    periods_per_crop = len(individual) // periods_per_year
    diversity_penalty = 0
    for i in range(0, len(individual), periods_per_crop):
        year_crops = [crop[0] for crop in individual[i:i+periods_per_crop]]
        unique_crops = len(set(year_crops))
        if unique_crops < periods_per_crop:
            diversity_penalty += (periods_per_crop - unique_crops) * 5
    unique_crops = len(set(crop[0] for crop in individual))
    impact_score = sum(impact_scores[crop[1]] for crop in individual)
    random_yield = sum(random.uniform(10, 30) for _ in individual)
    return unique_crops + impact_score + random_yield - diversity_penalty,

# Register GA functions to toolbox
toolbox.register("individual", create_individual, soil_type=None, num_periods=0, periods_per_year=0)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.4)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate, periods_per_year=0)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        soil_type = request.form.get("soil_type")
        num_periods = int(request.form.get("num_periods"))
        periods_per_year = int(request.form.get("periods_per_year"))

        # Update the toolbox with user inputs
        toolbox.register("individual", create_individual, soil_type=soil_type, num_periods=num_periods, periods_per_year=periods_per_year)
        toolbox.register("evaluate", evaluate, periods_per_year=periods_per_year)

        # Create population and run the GA
        population = toolbox.population(n=100)
        result_population, _ = algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.1, ngen=30, verbose=False)
        best_individual = tools.selBest(result_population, 1)[0]

        # Create DataFrame for result display
        result_df = pd.DataFrame(best_individual, columns=['Name', 'Impact', 'Soil_Type'])
        return render_template("index.html", result=result_df.to_html())

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
