# mega-heracross

Problem Statement 4 and the Bharatiya Antariksha Hackathon 2026 in general.

General Hackathon Information (BAH 2026)
Event Name: Bharatiya Antariksha Hackathon (BAH) 2026
Edition: 3rd Edition
Organizers: ISRO in collaboration with HackerSkill
Eligibility: Exclusive to students (Undergraduate, Postgraduate, PhD, Research Scholars). Professionals and non-students are not eligible.
Team Requirements: 3–4 members per team
Challenges: 15 total problem statements available (teams can choose any one)
Key Dates:
Registration & Submission Deadline: 1st July 2026
Shortlisting: ~28th July 2026
Grand Finale: A 30-hour offline event held at an ISRO center.
Platform: Hosted on the HackerSkill website.
Perks of Participating: Mentorship from ISRO scientists, potential internship opportunities at ISRO, national-level recognition, and networking with top student innovators.
Key Evaluation Criteria: INNOVATION is highlighted as the most important selection criteria.
Problem Statement 4 Details
Title: Route Resilience: Occlusion-Robust Road Extraction & Graph-Theoretic Criticality Analysis for Urban Mobility

Background & Problem
Modern urban centers (like rapidly expanding metropolises such as Bengaluru) face spatial modeling challenges—specifically fragmentation and stagnation. Standard satellite-based road extraction often fails due to "spectral blindness" caused by tree canopies, building shadows, and cloud cover. This results in "broken" masks that lack topological connectivity, making them useless for real-world applications like traffic simulation or disaster response.

Core Objective
To develop an operational, robust road extraction method and perform graph criticality analysis for urban mobility. The solution requires bridging the gap of spectral blindness by building an end-to-end pipeline broken down into two equal parts (a 50/50 balance between both):

Vision / Road Extraction: Using context-aware Deep Learning (such as transformer-based architectures) to "see through" occlusions (tree cover, building shadows) and extract robust road networks from satellite imagery.
Graph Criticality Analysis: Transforming these masks into a mathematically continuous, topologically connected, weighted graph to model urban mobility resilience. This involves quantifying network vulnerability through centrality metrics, identifying systemic bottlenecks, and building a framework to stress-test the network (simulating localized infrastructure failures or "urban collapse scenarios").
Evaluation Metrics
The prototype solutions for this problem statement will be evaluated based on:

Intersection over Union (IOU)
Dice scores
Model generalization
Connectivity ratio
Probabilistic accuracy
Maintaining an ideal 50% balance between road infrastructure accuracy and graph theoretical resilience analysis.
