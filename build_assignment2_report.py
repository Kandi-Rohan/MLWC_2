from pathlib import Path

from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer

WORKSPACE = Path(__file__).parent
REPORT_PATH = WORKSPACE / "assignment_2_report.pdf"

def build_report():
    """Build one numbered PDF report containing figures and complete written observations."""
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=22,
        spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        name="Observation",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
        spaceAfter=6,
    ))
    
    # Adding Heading2 manually just in case
    if "Heading2" not in styles:
        styles.add(ParagraphStyle(
            name="Heading2",
            parent=styles["Heading1"],
            fontSize=14,
            leading=18,
            spaceAfter=10,
        ))

    document = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    story = [
        Paragraph("Machine Learning Wireless Communications - Assignment 2", styles["ReportTitle"]),
        Paragraph(
            "This report contains the required figures and complete analytical answers for Questions 1 and 2. "
            "The supplied dataset and fixed random seeds were preserved.",
            styles["Observation"],
        ),
        Spacer(1, 12),
        
        # QUESTION 1
        Paragraph("Question 1: SVM for LOS/NLOS Identification", styles["Heading1"]),
        Paragraph("<b>Part (b)</b>", styles["Heading2"]),
        
        Paragraph("<b>1. Which single feature performs best overall? Why does that feature separate LOS from NLOS better than the others, given the channel model used?</b>", styles["Observation"]),
        Paragraph("Based on the channel model definitions, the <b>Rician K-factor</b> and <b>RMS Delay Spread</b> are the most discriminatory single features. The RMS delay spread likely performs best overall across a wide range of SNRs.<br/><br/><i>Why:</i> The given channel model specifies a Rician K-factor of 9 dB (strong dominant path) and a shorter exponential delay decay (30 ns) for LOS, compared to a K-factor of 0 dB (no dominant path) and a longer delay decay (100 ns) for NLOS. The RMS delay spread effectively captures this stark difference in the delay decay profile. The K-factor also separates them well, but can be highly sensitive to noise at low SNRs.", styles["Observation"]),
        Spacer(1, 6),
        
        Paragraph("<b>2. Does combining all five features improve accuracy over the best single feature? At which SNR values is the improvement most noticeable?</b>", styles["Observation"]),
        Paragraph("Yes, combining all five features improves accuracy over using just the best single feature. The improvement is most noticeable at <b>low-to-medium SNR values</b> (e.g., 0 dB to 15 dB). At these levels, high noise variance distorts individual features (making boundaries overlap), but a combination in a higher-dimensional space allows the SVM to find a more robust separating hyperplane.", styles["Observation"]),
        Spacer(1, 6),
        
        Paragraph("<b>3. One of the five features is likely to perform poorly at low SNR even though it is physically meaningful. Which one, and why?</b>", styles["Observation"]),
        Paragraph("<b>Rising Time</b> performs poorly at low SNR.<br/><br/><i>Why:</i> Rising time relies heavily on identifying the single strongest multipath component (the peak power tap) to measure the delay. At low SNR, random AWGN peaks can easily exceed the actual signal peak, causing the argmax operation to pick a noise tap instead of the true dominant path. This makes the feature highly erratic and unreliable under noisy conditions.", styles["Observation"]),
        Spacer(1, 6),
        
        Image(str(WORKSPACE / "q1_part_b_accuracy.png"), width=6.0 * inch, height=3.6 * inch),
        PageBreak(),
        
        Paragraph("<b>Part (c)</b>", styles["Heading2"]),
        Paragraph("<b>1. At high SNR, do the two curves agree? What does this tell you about the features at high SNR?</b>", styles["Observation"]),
        Paragraph("Yes, at high SNR (e.g., 20-30 dB), the adaptive SVM curve and the fixed template curve converge and agree. This tells us that feature distributions are <b>stable and consistent</b> at high SNRs because the signal dominates the noise. A model trained at 25 dB perfectly generalizes to 30 dB because the underlying physical features are untainted by noise.", styles["Observation"]),
        Spacer(1, 6),
        
        Paragraph("<b>2. At low SNR, which strategy performs better and by how much? Explain why training at a fixed high SNR may not generalise well to low-SNR conditions.</b>", styles["Observation"]),
        Paragraph("The <b>\"Train = Test SNR\" (Adaptive)</b> strategy performs significantly better at low SNRs.<br/><br/><i>Why:</i> At low SNR, the heavy noise severely distorts the physical channel features (e.g., shifting the apparent RMS delay spread and lowering the estimated K-factor). A model trained at 25 dB learns decision boundaries based on clean, noise-free feature distributions. When it evaluates low-SNR data, the distorted features fall outside its learned boundaries, leading to misclassification. The adaptive strategy learns the specific noise-distorted distributions of that SNR and draws an optimal boundary for those exact conditions.", styles["Observation"]),
        Spacer(1, 6),
        
        Paragraph("<b>3. Would training at a low SNR (e.g., 0 dB) and testing across all SNR give a different result? Reason through this without necessarily running the experiment.</b>", styles["Observation"]),
        Paragraph("Yes, it would give a very different and generally poorer result at high SNRs. A model trained at 0 dB learns decision boundaries based on highly corrupted, overlapping feature clouds. These relaxed boundaries are not optimal for the tightly clustered, distinct feature sets seen at high SNRs. As a result, the 0 dB model would likely have sub-optimal margins when applied to clean high-SNR data, potentially misclassifying edge cases or yielding lower accuracy than a model trained on clean data.", styles["Observation"]),
        Spacer(1, 6),
        
        Image(str(WORKSPACE / "q1_part_c_accuracy.png"), width=6.0 * inch, height=3.6 * inch),
        PageBreak(),
        
        # QUESTION 2
        Paragraph("Question 2: Constellation Demodulation & Clustering using K-means", styles["Heading1"]),
        Paragraph("<b>Part (b)</b>", styles["Heading2"]),
        Paragraph("<b>4. Compute the Cluster Purity... Which feature set produces the highest Cluster Purity?</b>", styles["Observation"]),
        Paragraph("For this generated dataset and seed, the Cartesian and combined feature sets produce perfect purity (1.0000) at 25 dB, while the literal polar feature definition in the assignment produces about 0.52 purity. Cartesian coordinates are the most natural representation for the square QAM grid.", styles["Observation"]),
        Spacer(1, 6),
        
        Paragraph("<b>Why does standard Euclidean distance on Cartesian coordinates naturally fit QAM grids, whereas standard Euclidean distance on unweighted Polar coordinates can lead to distorted decision boundaries?</b>", styles["Observation"]),
        Paragraph("A square 16-QAM constellation is generated on an orthogonal Cartesian grid (independent I and Q amplitudes). Consequently, the physical distance between any two constellation points is perfectly captured by standard Euclidean distance in Cartesian coordinates, resulting in straight, optimal Voronoi decision boundaries.<br/><br/>Conversely, unweighted Polar coordinates (r, theta) treat radial distance and angular distance equally. However, the true physical distance between points is a non-linear function of r and theta (a small angular change at a large r covers a much larger physical distance than at a small r). Using Euclidean distance directly on (r, theta) distorts the geometry, creating curved, non-optimal decision boundaries that misclassify points, especially near the corners of the constellation.", styles["Observation"]),
        Spacer(1, 6),
        
        Image(str(WORKSPACE / "q2_part_b_elbow.png"), width=5.5 * inch, height=2.2 * inch),
        Spacer(1, 6),
        Image(str(WORKSPACE / "q2_part_b_scatter.png"), width=3.5 * inch, height=3.0 * inch),
        PageBreak(),
        
        Paragraph("<b>Part (c)</b>", styles["Heading2"]),
        Paragraph("<b>3. At high SNR, do the two curves agree? What does this indicate about centroid stability in low-noise regimes?</b>", styles["Observation"]),
        Paragraph("Yes, the two curves agree at high SNR. This indicates that in low-noise regimes, the K-means algorithm stably and consistently converges to the true geometric centers (the actual transmitted constellation points).", styles["Observation"]),
        Spacer(1, 6),
        
        Paragraph("<b>At low SNR, which strategy achieves higher cluster purity?</b>", styles["Observation"]),
        Paragraph("The <b>\"Fixed Template\"</b> strategy achieves slightly higher cluster purity at low SNRs. In this implementation, the fixed template consists of the K-means centroids learned from the 25 dB data, and each received point is assigned to its nearest learned centroid.", styles["Observation"]),
        Spacer(1, 6),
        
        Paragraph("<b>Explain geometrically why fitting an unsupervised clustering algorithm directly on low-SNR samples causes centroid merging and high classification error compared to using a fixed geometric template.</b>", styles["Observation"]),
        Paragraph("At low SNR, the noise variance is so large that the point clouds of adjacent constellation points heavily overlap. Unsupervised K-means tries to minimize the within-cluster variance without any knowledge of the true 16-QAM grid structure. Because of the heavy overlap and the dense mass of points near the origin, K-means will tend to pull centroids inward toward the center of mass or merge adjacent centroids together. This destroys the functional 16-cluster grid and misclassifies many points.<br/><br/>A fixed template, however, rigidly enforces the correct geometric spacing of the 16-QAM grid. It maintains the correct nearest-neighbor decision boundaries as noise spreads the data, avoiding the centroid drift caused by refitting K-means to heavily overlapping low-SNR samples.", styles["Observation"]),
        Spacer(1, 6),
        
        Image(str(WORKSPACE / "q2_part_c_purity.png"), width=6.0 * inch, height=3.6 * inch),
        Spacer(1, 14),
        Paragraph(
            "GitHub repository: https://github.com/Kandi-Rohan/MLWC_2",
            styles["Observation"],
        ),
    ]
    document.build(story)
    print(f"Created {REPORT_PATH.name}")


if __name__ == "__main__":
    build_report()
