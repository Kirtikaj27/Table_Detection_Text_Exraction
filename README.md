# Table_Detection_Text_Exraction
```
DATA          : THIS FILE CONTAINS TRAINING AND TESTING DATA ALONG WITH THE OUTPUT IMAGES AND CSV FILES
	      	           EXTRACTED_TEXT    : CONTAINS OUTPUT CSV FILE AND EXTRACTED TEXT FILE
	                   MASKED_IMAGES     : CONTAINS EXTRACTED TABLE IMAGE
	                   TEST_DATA         : CONTAINS TEST DATA
	                   TEST_DATA_MASK    : CONTAINS GENERATED MASK FOR TEST DATA
	                   TRAINING_DATA     : CONTAINS TRAIN DATA

MODELS        : THIS FILE CONTAINS SAVED MODEL

SRC           : THIS FILE CONTAINS ALL THE CODE FILES
	                   S1_PREPROCESS     : CONTAINS FILE THAT GENERATES MASK FOR THE TRAINING DATA
	                   S2_PRODUCING_MASK : CONTAINS FILE FOR MODEL THAT DETECTS AND CREATE MASK FOR IMAGES
	                   S3_APPLYING_MAKS  : CONTAINS FILE FOR EXTRACTING TABLE USING ORIGINAL IMAGE AND GENERATED MASK
	                   S4_TEXTEXTRACT    : CONTAINS FILE THAT TAKES EXTRACTED TABLE IMAGE AND GENERATES CSV FILE

CONFIG        : CONTAINS REQUIRED FOLDER PATHS

TESSERACT-OCR : INSTALLATION DIRECTORY FOR TESERRACT-OCR

MAIN          : FILE CONTANING REQUIRED FILEPATHS AND FUNCTION CALLS


RUNNING THE PROGRAM :
STEP 1 : DOWNLOAD MARMOT DATASET AND PUT IN TRAINING_DATA
STEP 2 : PUT IMAGE TO TEST IN THE TEST_DATA FOLDER
STEP 3: REPLACE THE FILENAME IN MAIN.PY FILE WITH THE NAME OF TEST IMAGE
STEP 4: RUN THE MAIN.PY FILE FROM CMD.
STEP 5: CHECK THE DETECTED TABLE IN THE MASKED_IMAGES FOLDER
STEP 6: CHECK THE CSV FILE IN THE EXTRACTED_TEXT FOLDER

```

APPROACH TAKEN : First thing was to detect the tables in the images for which a model is built based on the TableNet model architecture. The model is trained on 2500 epochs. This model masks the image in the area where the table is detected. Next, the mask image and original image is put together highlighting area where the table is present and blacking out the other area. This way we get the the image of the table. Last part is to read text from the table and convert it into a csv file.

IMPROVEMENTS :
Data Augmentation for better results. Also, train the model for more number of epochs.
Using a better approach for text extraction and conversion to csv file.

REFERENCE : 
https://github.com/jainammm/TableNet
