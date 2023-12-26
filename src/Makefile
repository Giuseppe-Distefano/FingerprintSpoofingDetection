clean_files:
	rm -rf ../output/FeaturesAnalysis/Histograms/*.png || true
	rm -rf ../output/FeaturesAnalysis/Heatmaps/*.png || true
	rm -rf ../output/DimensionalityReduction/PCA/*.png || true
	rm -rf ../output/DimensionalityReduction/LDA/*.png || true
	rm -rf ../output/Training/*.txt || true
	rm -rf ../output/Training/*.csv || true
	rm -rf ./modules/__pycache__ || true
	rm ./tmp.txt || true
	clear

clean_all:
	make clean_files
	rm -rf ../output/FeaturesAnalysis/Histograms/ || true
	rm -rf ../output/FeaturesAnalysis/Heatmaps/ || true
	rm -rf ../output/DimensionalityReduction/PCA/ || true
	rm -rf ../output/DimensionalityReduction/LDA/ || true
	rm -rf ../output/FeaturesAnalysis/ || true
	rm -rf ../output/DimensionalityReduction/ || true
	rm -rf ../output/Training/ || true

setup:
	mkdir ../output/FeaturesAnalysis || true
	mkdir ../output/FeaturesAnalysis/Histograms || true
	mkdir ../output/FeaturesAnalysis/Heatmaps || true
	mkdir ../output/DimensionalityReduction || true
	mkdir ../output/DimensionalityReduction/PCA || true
	mkdir ../output/DimensionalityReduction/LDA || true
	mkdir ../output/Training || true

run:
	python3 main.py
	