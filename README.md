# US Wildfire Risk Map (FireTwin-US)

Interactive wildfire risk and historical fire analog visual analytics prototype for CSE 6242 / CX 4242.

## Live Links
- **Interactive Dashboard Prototype**: [https://ramonqu7.github.io/us-wildfire-risk-map/](https://ramonqu7.github.io/us-wildfire-risk-map/)
- **Interactive Proposal Slide Deck (with Embedded Dashboard)**: [https://ramonqu7.github.io/us-wildfire-risk-map/slides.html](https://ramonqu7.github.io/us-wildfire-risk-map/slides.html)
- **Editable Online Wireframe (Draw.io / diagrams.net)**: [Open in diagrams.net](https://app.diagrams.net/?#Uhttps%3A%2F%2Fraw.githubusercontent.com%2Framonqu7%2Fus-wildfire-risk-map%2Fmain%2Fdocs%2Fdashboard_wireframe.drawio)

## Repository Structure
- `docs/index.html` — Interactive dashboard prototype (Leaflet, bounded lightness choropleth, crisp 80% analog contours, GMM soft regime bar, What-If sliders, and top-k historical fire twin cards).
- `docs/slides.html` — Interactive 2-minute presentation slide deck with embedded live dashboard.
- `docs/dashboard_wireframe.drawio` — Editable wireframe sketch for Task B.
- `poc/run_poc.py` — Local proof-of-concept clustering (GMM) and feature-weighted Case-Based Reasoning (CBR) scripts over FPA-FOD, FEMA NRI, and USFS WRC datasets.
