AI-Driven Electrochemical Image Generation and Classification<br>
This repository contains two parts of the software used in the research paper "Artificial Intelligence-assisted electrochemical sensors for qualitative and semi-quantitative multiplexed analyses". The software pipeline is split into two main components:<br>
<br>
Java Module: Responsible for generating images from voltammetric data.<br>
Python Module: This module is used to classify the generated images and identify pollutants.<br>

Prerequisites<br>
Java 21 or higher<br>
Python 3.12 or higher<br>

Required Python libraries listed in requirements.txt<br>
<br>
Structure<br>
java/: Contains the Java code for generating images from voltammetric sequences.<br>
python/: Contains the Python code for image classification using machine learning models.<br>
data/: Folder where input voltammetric data and generated images are stored.<br>
<br>
Java Module: Image Generation<br>
The Java module converts voltammetric data into images using Gramian Angular Field (GAF) transformations. These images serve as input for the classification module. To run the Java module, navigate to the java/ directory and execute the following command:<br>
<br>
gaf_generation.sh<br>
Output images will be saved in the data/images/ folder.<br>
<br>
Python Module: Image Classification<br>
The Python module uses a Convolutional Neural Network (CNN) to classify the images generated from the voltammetric sequences. The classification is based on identifying the presence of specific pollutants in the sample.<br>
<br>
To set up the Python environment, run:<br>
<br>
pip install -r requirements.txt<br>
Then, to run the classification:<br>
<br>
gaf_cnn.sh<br>
The results will be shown on the screen, including classification accuracy and pollutant identification.<br>
<br>
Dataset<br>
The dataset used for training and validation consists of voltammetric sequences transformed into images. The images are stored in the data/ folder and classified according to the pollutants detected.<br>
<br>
Citation<br>
If you use this software for your research, please cite our paper:<br>
<br>
Rocco Cancelliere, Mario Molinara, Antonio Maffucci, Laura Micheli, "Artificial Intelligence-assisted electrochemical sensors for qualitative and semi-quantitative multiplexed analyses."<br>
<br>
Complete Output Files<br>
The original gaf_cnn.py prints its results (probabilities, confusion matrix, accuracy, classification report) on the screen only. To keep every generated result, use:<br>
<br>
python/gaf_cnn_full_output.sh<br>
<br>
which runs the same trained model (CNN_5classes__epo_4000_pat_1000_batch_16_tl.keras) on both test sets and writes all outputs into the results/ folder:<br>
<br>
results/CNN_COMPLETE_OUTPUT.txt - single complete text file: overall results, model architecture, the full probability matrix for every image, per image predictions, confusion matrices, accuracy (overall and per class), classification report and the list of misclassified images.<br>
results/CNN_FULL_REPORT.html - the same results as a self contained HTML report (open in a browser).<br>
results/&lt;split&gt;_predictions.csv - one row per test image with the probability of each of the 5 classes.<br>
results/&lt;split&gt;_confusion_matrix.csv / .png - confusion matrix as table and as image.<br>
results/&lt;split&gt;_classification_report.txt - precision / recall / f1 report.<br>
results/&lt;split&gt;_summary.json and all_splits_summary.json - machine readable summaries.<br>
results/all_generated_images_manifest.csv - inventory of all 264 generated PNG images in data/ (CNN input sets and the raw GAF generator output).<br>
<br>
Final Result Report<br>
The complete results are also available as a standalone report in three formats:<br>
<br>
results/CNN_Final_Report.pdf - PDF report (A4, ready to print or share).<br>
results/CNN_Final_Report.docx - the same report as an editable Word document.<br>
results/CNN_Final_Report.txt - the same report as plain text.<br>
<br>
<br>
Jupyter Notebook<br>
The whole pipeline (loading the trained model, running it on the test sets, all tables, confusion matrices, per image predictions, the final report and download links for every file) is available as a single notebook in the repository root:<br>
<br>
GAF_CNN_Final_Report.ipynb<br>
<br>
Open it in Jupyter and run all cells. It needs the libraries listed in python/requirements.txt only, no training is performed. At the end it builds results/CNN_Final_Report.pdf, .docx, .txt and packs everything into results/CNN_Results_Package.zip.<br>
<br>
<br>
Single folder with everything (GAF_CNN_Complete)<br>
GAF_CNN_Complete/ is one self contained folder holding the whole project: the code, the data, the results and the notebooks. It is a copy of the repository content plus the outputs of the gaf_cnn.sh run, so it can be downloaded and read on its own:<br>
<br>
GAF_CNN_Complete/code/ - python and java sources (including run_support/sitecustomize.py, the capture layer used during the run)<br>
GAF_CNN_Complete/data/ - the voltammetric CSVs and all GAF images<br>
GAF_CNN_Complete/results/ - every report, table, figure and the ZIP package<br>
GAF_CNN_Complete/notebooks/ - all notebooks, executed, with their outputs<br>
GAF_CNN_Complete/run_output/gaf_cnn_sh/ - the outputs of the gaf_cnn.sh run (console log, training history, plots, trained model, test evaluation)<br>
GAF_CNN_Complete/original_model/ - the model that was in the repository before the run<br>
<br>
Run of gaf_cnn.sh and its notebooks<br>
gaf_cnn.sh was executed end to end (training 4000 epochs with early stopping, then the test phase). Every output of that run is stored in results and in GAF_CNN_Complete/run_output/gaf_cnn_sh/, and each output is readable in its own notebook:<br>
<br>
GAF_CNN_Run_01_Training_Output.ipynb - training log, curves, epoch table, trained model<br>
GAF_CNN_Run_02_Test_Output.ipynb - predictions, confusion matrix, accuracy, classification report<br>
GAF_CNN_Run_03_All_Outputs.ipynb - index of all output files with download links<br>
<br>
Scripts used: python/finalize_run.sh (collect the run output), python/make_run_notebooks.py (build the notebooks), python/execute_notebooks.py (execute notebooks), python/build_results_zip.py (package the results).<br>
<br>
The report contains: model and dataset summary, overall accuracy, accuracy per class, prediction confidence, confusion matrices (as tables and images), classification reports, per image predictions, misclassified images, conclusion and an appendix with the full probability matrix. It is generated by python/make_final_report.py (requires python-docx and reportlab, see python/requirements-report.txt).<br>

