from src.s1_preprocess.xml_jpeg_mask import *
from src.s2_producing_mask.generate_mask_main import *
from src.s3_applying_mask.image_mask import *
from src.s4_textextract.text_extract import *

import configparser
import os

def parsing_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    training_data_folder        = config['Data']['training_data_folder']
    test_data_folder            = config['Data']['test_data_folder']
    test_data_mask_folder       = config['Data']['test_data_mask_folder']
    test_data_masked            = config['Data']['test_data_masked']
    output_text                 = config['Data']['output_text']
    model_folder_path           = config['Model']['model_path']
    tesseract_path              = config['Tesseract']['tesseract_path']
    return training_data_folder, test_data_folder, test_data_mask_folder, test_data_masked, output_text, tesseract_path, model_folder_path

training_data_folder, test_data_folder, test_data_mask_folder, test_data_masked, output_text, tesseract_path, model_folder_path = parsing_config()

marmot_data = training_data_folder+"\\Marmot_data"
column_data = training_data_folder+"\\column_mask"
table_data  = training_data_folder+"\\table_mask"

input_train_images_path     = training_data_folder+"\\Marmot_data" + "\\*.jpg"
model_path                  = model_folder_path + "\\model_final_1.pb"

filename = "test.jpg"
org_test_img_path           = test_data_folder + "\\\\" + filename
mask_img_path               = test_data_mask_folder + "\\\\" + filename.split('.')[0] + "_mask.jpeg"

test_file_mask_imposed_path = test_data_masked + "\\\\" + filename
output_csv_path           = output_text + "\\\\" + filename.split('.')[0] + ".csv"
output_text_path           = output_text + "\\\\" + filename.split('.')[0] + ".txt"

# convert_xml_jpeg_train(marmot_data, column_data, table_data)
# # # train(input_train_images_path, model_path)
# test(model_path, org_test_img_path, mask_img_path)
# impose_mask(org_test_img_path, mask_img_path, test_file_mask_imposed_path) output_excel_path, 
text_extract(test_file_mask_imposed_path, output_csv_path, tesseract_path, output_text_path)
