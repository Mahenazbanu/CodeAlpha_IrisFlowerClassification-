"""
Iris Flower Classification - Gradio Web Interface

A modern, user-friendly web application for predicting Iris flower species
using trained machine learning models.

Features:
- Interactive input fields for flower measurements
- Real-time prediction with confidence scores
- Input validation and error handling
- Responsive, mobile-friendly design
- Model information display

Author: ML Engineer
Date: 2026-06-14
"""

import os
import warnings
from typing import Dict, Any, Tuple, Optional

import gradio as gr
import numpy as np
from joblib import load

# Suppress sklearn warnings about feature names
warnings.filterwarnings('ignore', category=UserWarning)


class IrisPredictor:
    """
    Prediction handler for Iris flower classification.

    Loads the trained model and provides prediction functionality
    with proper validation and error handling.
    """

    def __init__(self, model_path: str = "iris_classifier.pkl") -> None:
        """
        Initialize the predictor by loading the saved model.

        Args:
            model_path: Path to the saved model pickle file.
        """
        self.model_path = model_path
        self.model_data = None
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.model_name = None
        self.load_model()

    def load_model(self) -> None:
        """Load the trained model, scaler, and label encoder from disk."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Please run train_model.py first to train and save the model."
            )

        self.model_data = load(self.model_path)
        self.model = self.model_data['model']
        self.scaler = self.model_data['scaler']
        self.label_encoder = self.model_data['label_encoder']
        self.model_name = self.model_data['model_name']

    def validate_inputs(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float
    ) -> Optional[str]:
        """
        Validate user input values for flower measurements.

        Args:
            sepal_length: Sepal length in cm.
            sepal_width: Sepal width in cm.
            petal_length: Petal length in cm.
            petal_width: Petal width in cm.

        Returns:
            Error message string if validation fails, None otherwise.
        """
        # Check for None or empty values
        inputs = {
            'Sepal Length': sepal_length,
            'Sepal Width': sepal_width,
            'Petal Length': petal_length,
            'Petal Width': petal_width,
        }

        for name, value in inputs.items():
            if value is None or value == '':
                return f"Error: {name} is required. Please enter a value."

        # Check for valid numeric ranges (realistic biological ranges)
        validations = [
            (sepal_length, 'Sepal Length', 0.1, 30.0),
            (sepal_width, 'Sepal Width', 0.1, 30.0),
            (petal_length, 'Petal Length', 0.1, 30.0),
            (petal_width, 'Petal Width', 0.1, 30.0),
        ]

        for value, name, min_val, max_val in validations:
            if not isinstance(value, (int, float)):
                return f"Error: {name} must be a number."
            if value <= 0:
                return f"Error: {name} must be greater than 0."
            if value < min_val or value > max_val:
                return f"Error: {name} must be between {min_val} and {max_val} cm."

        return None

    def predict(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float
    ) -> Dict[str, Any]:
        """
        Predict the Iris species from flower measurements.

        Args:
            sepal_length: Sepal length in cm.
            sepal_width: Sepal width in cm.
            petal_length: Petal length in cm.
            petal_width: Petal width in cm.

        Returns:
            Dictionary with prediction results.
        """
        # Create feature array
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]

        # Decode label
        species = self.label_encoder.inverse_transform([prediction])[0]
        confidence = probabilities[prediction] * 100

        # Get all class probabilities
        all_probs = {
            self.label_encoder.inverse_transform([i])[0]: round(prob * 100, 2)
            for i, prob in enumerate(probabilities)
        }

        return {
            'species': species,
            'confidence': confidence,
            'all_probabilities': all_probs,
        }


# Initialize predictor
try:
    predictor = IrisPredictor("iris_classifier.pkl")
    MODEL_LOADED = True
except FileNotFoundError as e:
    predictor = None
    MODEL_LOADED = False
    MODEL_ERROR = str(e)


def classify_iris(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float
) -> Tuple[str, str, str, str]:
    """
    Gradio interface function for Iris classification.

    Args:
        sepal_length: Sepal length in cm.
        sepal_width: Sepal width in cm.
        petal_length: Petal length in cm.
        petal_width: Petal width in cm.

    Returns:
        Tuple of (species_output, confidence_output, probabilities_output, model_output).
    """
    # Check if model is loaded
    if not MODEL_LOADED or predictor is None:
        return (
            "⚠️ Model Error",
            "N/A",
            f"Failed to load the model.\n\n{MODEL_ERROR}\n\n"
            "Please run 'python train_model.py' first to train the model.",
            "N/A"
        )

    # Validate inputs
    error_msg = predictor.validate_inputs(
        sepal_length, sepal_width, petal_length, petal_width
    )
    if error_msg:
        return (
            "⚠️ Validation Error",
            "N/A",
            error_msg,
            predictor.model_name if predictor else "N/A"
        )

    try:
        # Perform prediction
        result = predictor.predict(
            sepal_length, sepal_width, petal_length, petal_width
        )

        # Format species name for display
        species_clean = result['species'].replace('Iris-', '').title()
        species_display = f"🌸 Iris {species_clean}"

        # Format confidence
        confidence_display = f"{result['confidence']:.1f}%"

        # Format all probabilities
        probs_text = "### Class Probabilities\n\n"
        for class_name, prob in sorted(
            result['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            bar_length = int(prob / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            clean_name = class_name.replace('Iris-', '').title()
            marker = " ✓" if class_name == result['species'] else ""
            probs_text += f"**{clean_name}**: {bar} {prob:.1f}%{marker}\n\n"

        model_display = f"{predictor.model_name}"

        return species_display, confidence_display, probs_text, model_display

    except Exception as e:
        return (
            "⚠️ Prediction Error",
            "N/A",
            f"An error occurred during prediction:\n\n{str(e)}",
            predictor.model_name if predictor else "N/A"
        )


def clear_inputs() -> Tuple[float, float, float, float, str, str, str, str]:
    """
    Clear all input fields and output displays.

    Returns:
        Tuple of empty/default values for all components.
    """
    return 0.0, 0.0, 0.0, 0.0, "", "", "", ""


# Custom CSS for a modern, nature-inspired design
CUSTOM_CSS = """
.iris-header {
    text-align: center;
    padding: 20px 0;
    background: linear-gradient(135deg, #E65100 0%, #FF8F00 100%);
    color: #FFFFFF;
    border-radius: 12px;
    margin-bottom: 20px;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
}
.iris-title {
    font-size: 2em;
    font-weight: bold;
    margin-bottom: 5px;
    color: #FFFFFF;
}
.iris-subtitle {
    font-size: 1.1em;
    color: #FFE0B2;
}
.input-section {
    background-color: #1E1E1E;
    border: 2px solid #FF8F00;
    border-radius: 12px;
    padding: 15px;
    color: #FFFFFF;
}
.output-section {
    background-color: #1E1E1E;
    border: 2px solid #FF8F00;
    border-radius: 12px;
    padding: 15px;
    color: #FFFFFF;
}
.result-box {
    font-size: 1.5em;
    font-weight: bold;
    text-align: center;
    padding: 15px;
    border-radius: 8px;
    background: linear-gradient(135deg, #E65100 0%, #FF8F00 100%);
    color: #FFFFFF;
    border: 2px solid #FF8F00;
    text-shadow: 1px 1px 3px rgba(0,0,0,0.5);
}
.confidence-box {
    font-size: 1.3em;
    text-align: center;
    padding: 12px;
    border-radius: 8px;
    background-color: #2C2C2C;
    color: #FFB74D;
    border: 2px solid #FF8F00;
    font-weight: 600;
}
.model-info {
    text-align: center;
    font-size: 0.9em;
    color: #FFB74D;
    padding: 8px;
    background-color: #2C2C2C;
    border-radius: 6px;
    border: 1px solid #FF8F00;
}
.footer-text {
    text-align: center;
    font-size: 0.85em;
    color: #FFCC80;
    margin-top: 15px;
    padding: 10px;
}
"""


def create_interface() -> gr.Blocks:
    """
    Create the Gradio web interface for Iris classification.

    Returns:
        Gradio Blocks interface.
    """


    with gr.Blocks() as demo:

        # Header
        gr.HTML("""
        <div class="iris-header">
            <div class="iris-title">🌺 Iris Flower Classification</div>
            <div class="iris-subtitle">Machine Learning-Based Species Predictor</div>
        </div>
        """)

        with gr.Row():
            # Input Section
            with gr.Column(scale=1):
                gr.HTML('<div class="input-section">')
                gr.Markdown("### 📏 Enter Flower Measurements (cm)")

                sepal_length = gr.Number(
                    label="Sepal Length",
                    value=5.1,
                    minimum=0.1,
                    maximum=30.0,
                    step=0.1,
                    info="Length of the sepal in centimeters"
                )
                sepal_width = gr.Number(
                    label="Sepal Width",
                    value=3.5,
                    minimum=0.1,
                    maximum=30.0,
                    step=0.1,
                    info="Width of the sepal in centimeters"
                )
                petal_length = gr.Number(
                    label="Petal Length",
                    value=1.4,
                    minimum=0.1,
                    maximum=30.0,
                    step=0.1,
                    info="Length of the petal in centimeters"
                )
                petal_width = gr.Number(
                    label="Petal Width",
                    value=0.2,
                    minimum=0.1,
                    maximum=30.0,
                    step=0.1,
                    info="Width of the petal in centimeters"
                )

                with gr.Row():
                    predict_btn = gr.Button(
                        "🔍 Predict Species",
                        variant="primary",
                        size="lg",
                        scale=3
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary",
                        size="lg",
                        scale=1
                    )

                gr.HTML('</div>')

                # Reference information
                with gr.Accordion("📊 Dataset Reference Ranges", open=False):
                    gr.Markdown("""
                    **Iris-setosa:**
                    - Sepal Length: 4.3 - 5.8 cm
                    - Sepal Width: 2.3 - 4.4 cm
                    - Petal Length: 1.0 - 1.9 cm
                    - Petal Width: 0.1 - 0.6 cm

                    **Iris-versicolor:**
                    - Sepal Length: 4.9 - 7.0 cm
                    - Sepal Width: 2.0 - 3.4 cm
                    - Petal Length: 3.0 - 5.1 cm
                    - Petal Width: 1.0 - 1.8 cm

                    **Iris-virginica:**
                    - Sepal Length: 4.9 - 7.9 cm
                    - Sepal Width: 2.2 - 3.8 cm
                    - Petal Length: 4.5 - 6.9 cm
                    - Petal Width: 1.4 - 2.5 cm
                    """)

            # Output Section
            with gr.Column(scale=1):
                gr.HTML('<div class="output-section">')
                gr.Markdown("### 🎯 Prediction Results")

                species_output = gr.Textbox(
                    label="Predicted Species",
                    value="",
                    interactive=False,
                    elem_classes="result-box"
                )

                confidence_output = gr.Textbox(
                    label="Confidence Score",
                    value="",
                    interactive=False,
                    elem_classes="confidence-box"
                )

                probabilities_output = gr.Markdown(
                    value="### Class Probabilities\n\nEnter values and click Predict to see results."
                )

                model_output = gr.Textbox(
                    label="Model Used",
                    value=predictor.model_name if predictor else "Loading...",
                    interactive=False,
                    elem_classes="model-info"
                )

                gr.HTML('</div>')

        # Footer
        gr.HTML("""
        <div class="footer-text">
            <p>Built with ❤️ using Python, scikit-learn, and Gradio</p>
            <p>Trained on the classic Iris dataset (Fisher, 1936) • 5 ML models compared</p>
        </div>
        """)

        # Event handlers
        predict_btn.click(
            fn=classify_iris,
            inputs=[sepal_length, sepal_width, petal_length, petal_width],
            outputs=[species_output, confidence_output, probabilities_output, model_output]
        )

        clear_btn.click(
            fn=clear_inputs,
            inputs=[],
            outputs=[
                sepal_length, sepal_width, petal_length, petal_width,
                species_output, confidence_output, probabilities_output, model_output
            ]
        )

    return demo


def main() -> None:
    """Launch the Gradio web interface."""
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CUSTOM_CSS
    )


if __name__ == "__main__":
    main()