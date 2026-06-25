#!/bin/bash
echo "🛰️ Building NeuroGrid container..."
docker build -t neurogrid-partc .
echo ""
echo "🔍 Running headless analysis..."
docker run --rm \
    -v $(pwd)/part_c_resilience/outputs:/app/part_c_resilience/outputs \
    neurogrid-partc /usr/local/bin/headless_entrypoint.sh
echo ""
echo "📁 Output artifacts:"
ls -lh part_c_resilience/outputs/disaster_heatmap.html part_c_resilience/outputs/evaluation.json 2>/dev/null || echo "(files not found)"
echo ""
echo "📄 To view the interactive dashboard, run:"
echo "   docker-compose up"
echo "   then open http://localhost:8501"
