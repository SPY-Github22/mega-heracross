# Mega-Heracross: Technical Appendix & Prototype Validation
**ISRO BAH 2026 - Problem Statement 4 (Route Resilience)**

## 1. Executive Summary & USP
Mega-Heracross goes beyond traditional static disaster mapping. It is a fully automated, end-to-end pipeline that takes fused optical and SAR (Synthetic Aperture Radar) satellite imagery, extracts a topological road network, and performs dynamic graph-theory resilience analysis. 

**Our Unique Selling Proposition (USP):**
Most disaster response systems rely on pre-existing, static road maps that become obsolete the moment a disaster strikes (e.g., floods washing away roads). Our prototype proves that we can dynamically extract the *current* state of the road network from space and immediately calculate the cascading effects of chokepoint failures using Monte Carlo simulations.

## 2. Prototype Architecture & Workflow
Our prototype was built and validated on a fully functional Python pipeline consisting of three core engines:
1. **Part A (Vision Engine):** A custom 12-channel SegFormer B3 neural network trained to extract road masks from fused LISS-IV (Optical) and Sentinel-1 (SAR) imagery.
2. **Part B (Skeleton Engine):** A morphological extraction engine that converts the pixel mask into a routable mathematical graph (nodes and edges) using Zhang-Suen skeletonization.
3. **Part C (Resilience Engine):** A graph-theoretic analyzer that calculates Brandes Betweenness Centrality for every intersection and runs 1,000 Monte Carlo simulations to model cascading traffic failures if key nodes collapse.

## 3. Real-World Validation (Koramangala, Bengaluru)
We didn't just build a theory—we ran our pipeline against the real road network of Koramangala, Bengaluru, extracted from OpenStreetMap (OSMnx).

### 3.1 Model Accuracy (Computer Vision)
To ensure our model generalizes across different urban layouts, we trained our SegFormer B3 on 10 topologically diverse regions of Bengaluru. 
* **Node F1 Score:** 0.4408 (A 26x improvement over our baseline model).
* **Edge F1 Score:** 0.3282 (With a precision of 0.8117, meaning when our model predicts a road connection, it is highly accurate).

*[Insert: training_curve.png here]*

### 3.2 Network Resilience Results (Graph Theory)
Running our Part C Engine on the 1,734 nodes and 2,307 edges of Koramangala yielded the following disaster resilience metrics:
* **Top Critical Chokepoint:** Node ID 11809900885 (Centrality Score: 1.000). If this intersection fails, it forces the maximum number of emergency detours across the entire neighborhood.
* **Network Collapse Threshold:** It requires the targeted removal of exactly 71 critical nodes to induce a 51.7% efficiency penalty on the network.
* **Cascading Failure:** During a localized disaster, the median cascade length is 10 nodes, with a failure acceleration rate of 0.32x.
* **Overall Monte Carlo Survival Rate:** 99.1% (Tested across 1,000 disaster simulations).

## 4. Conclusion & Impact
Mega-Heracross provides disaster response teams (like NDRF or BBMP) with a live, dynamic "heatmap" of network criticality. By knowing exactly which intersections bear the highest routing load during a crisis, emergency services can pre-position assets, defend critical bridges, or establish bypasses *before* the network collapses.

*[Insert: Full screen screenshot of disaster_heatmap_real_osmnx.html here]*
