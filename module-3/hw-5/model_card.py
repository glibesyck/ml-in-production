from model_card_toolkit import ModelCardToolkit
import os

output_dir = "./model_card"

mct = ModelCardToolkit(output_dir)

model_card = mct.scaffold_assets()

model_card.model_details.name = "Simple CNN for Dead Leaves Images Membership Classification"
model_card.model_details.overview = "This model distinguishes images of dead leaves (circles of different radius put on top of each other) on the matter of membership classification. That being said, it predicts whether points of coordinates [300, 512] and [700, 512] of original 1024x1024 images are on the same circle ('same' class) of on different ones ('different' class). Dataset is generated customly and it uses 1370 images for training and 684 for validation (perfectly balanced classes)."

model_card.quantitative_analysis.performance_metrics = [{
    "type": "accuracy",
    "value": 0.78
}]

# Limitations
model_card.considerations.limitations = [
    "This model is specific for the use case - it performs well on the variety of the images but please remember that it tries to identify whether 2 points are related to the same object and coordinates of these points are fixed, you can't change them."
]


mct.update_model_card(model_card)
mct.export_format()
