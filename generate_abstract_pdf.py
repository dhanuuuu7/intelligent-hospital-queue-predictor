import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

def build_abstract_pdf(filename: str):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=0.8 * inch,
        rightMargin=0.8 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch
    )

    styles = getSampleStyleSheet()

    # Title style: Centered, Bold 18pt
    title_style = ParagraphStyle(
        name='AbstractTitle',
        fontName='Times-Bold',
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        spaceAfter=18
    )

    # Body style: Justified, 11pt, 15pt leading
    body_style = ParagraphStyle(
        name='AbstractBody',
        fontName='Times-Roman',
        fontSize=11,
        leading=15.5,
        alignment=TA_JUSTIFY,
        spaceAfter=16
    )

    # Keywords style: Left aligned, 11pt
    keywords_style = ParagraphStyle(
        name='KeywordsStyle',
        fontName='Times-Roman',
        fontSize=11,
        leading=15,
        alignment=TA_LEFT,
        spaceAfter=18
    )

    # Roll numbers label style
    roll_header_style = ParagraphStyle(
        name='RollHeaderStyle',
        fontName='Times-Bold',
        fontSize=11,
        leading=15,
        alignment=TA_LEFT,
        spaceAfter=10
    )

    # Roll numbers item style
    roll_item_style = ParagraphStyle(
        name='RollItemStyle',
        fontName='Times-Roman',
        fontSize=11,
        leading=16,
        alignment=TA_LEFT,
        spaceAfter=5
    )

    # Faculty signature style: Right aligned
    signature_style = ParagraphStyle(
        name='SignatureStyle',
        fontName='Times-Bold',
        fontSize=11,
        leading=15,
        alignment=TA_RIGHT,
        spaceBefore=16
    )

    story = []

    # Title
    story.append(Paragraph("Abstract", title_style))

    # Abstract Body Text
    abstract_text = (
        "The Intelligent Hospital Queue & Appointment Optimization System is designed to automatically "
        "predict patient waiting times, estimate empirical uncertainty intervals, detect operational bottlenecks, "
        "and provide real-time decision support for hospital outpatient departments (OPDs). With the rapid growth "
        "of hospital patient inflow and unpredictable emergency cases, manual queue management leads to severe "
        "overcrowding, patient anxiety, and inefficient doctor allocation. The proposed system utilizes Discrete-Event "
        "Simulation (DES) using SimPy to model realistic patient encounters across multiple clinical departments, "
        "capturing arrival dynamics, doctor availability, triage delays, and emergency preemptions. Data preprocessing "
        "techniques, including chronological train/validation/test splitting and ColumnTransformer scaling/encoding, "
        "are combined with domain feature engineering (such as effective doctor load, traffic momentum, emergency "
        "pressure, and scheduled ratio ahead) to extract high-leverage predictive signals. Machine learning algorithms, "
        "primarily a tuned Random Forest Regressor alongside XGBoost, are deployed to predict waiting times, while "
        "department-specific quantile residual calibration provides reliable empirical 90% confidence intervals. "
        "The system incorporates a FastAPI backend integrated with a WebSockets push event bus for zero-polling real-time "
        "updates and an SQLAlchemy relational database. Role-based portals distinguish between staff operational controls "
        "(capacity management, \"What-If\" scenario simulation, bottleneck forecasting) and patient read-only observer "
        "views (OP key identity verification, ticket tracking). Furthermore, a continuous MLOps feedback loop tracks "
        "live performance metrics (MAE, RMSE, PSI drift) and triggers autonomous zero-downtime model retraining when "
        "performance drifts. The system provides a faster, automated, and data-driven approach for hospital operations "
        "management, significantly reducing patient wait times and administrative burden."
    )
    story.append(Paragraph(abstract_text, body_style))

    # Keywords Section
    keywords_text = (
        "<b>Keywords:</b> Intelligent Hospital Queue System, Discrete-Event Simulation (SimPy), "
        "Random Forest Regressor, Feature Engineering, Prediction Intervals, FastAPI, WebSockets, "
        "MLOps, Role-Based Access Control, Data Visualization."
    )
    story.append(Paragraph(keywords_text, keywords_style))

    # Roll Numbers Section
    story.append(Paragraph("<b>Roll Numbers :</b>", roll_header_style))
    
    roll_numbers = [
        "24B11CS299 :",
        "24B11CS128 :",
        "24B11CS520 :",
        "24B11CS198 :"
    ]
    for roll in roll_numbers:
        story.append(Paragraph(roll, roll_item_style))

    # Faculty Signature Line
    story.append(Paragraph("<b>Signature Of the Faculty :</b>", signature_style))

    doc.build(story)
    print(f"Successfully generated 1-page PDF at: {filename}")

if __name__ == "__main__":
    out_pdf = os.path.join(os.path.dirname(__file__), "abstract_submission.pdf")
    build_abstract_pdf(out_pdf)
