# Driver's License Recognition Project
<h2>Overview</h2>
This project aims to recognize and process driver's licenses issued in the United States. The task involves sorting through a multitude of documents, including driver's licenses and other types of documents. The objective is to identify all driver's licenses, extract text from them, and compile the information into a unified table.
<hr>
<h2>Technologies Used</h2>

- ResNet for Document Filtering: Utilized ResNet for document filtering to distinguish driver's licenses from other documents and images.
- YOLO8 and TrOCR for Text Recognition: Implemented YOLO8 and Trocr for field recognition on the documents and extraction of text.

<h3>Additional ResNet Models:</h3>

- State Classification: Employed an additional ResNet model for classifying the issuing state of the driver's license.
- Document Orientation Detection: Utilized another ResNet model to determine the orientation of the document (whether it's upright or upside-down) for proper alignment.

<h2>Workflow</h2>

- Document Filtering: ResNet model is used to filter out driver's licenses from the pool of documents.

- Text Recognition: YOLO8 and Trocr are employed to recognize fields on the driver's licenses and extract text from them.
  
- State Classification: An additional ResNet model is used to classify the state of issuance of the driver's license.
  
- Document Orientation Correction: Another ResNet model is utilized to detect the orientation of the document and correct it if necessary.
  
- PDF to JPG Conversion: All PDF files are converted to JPG format for easier processing.
  
<h2>Usage</h2>
To use this project:

Ensure you have the necessary dependencies installed.
Place all documents in the designated input directory.
Run the main script to initiate the recognition and processing pipeline.
Retrieve the output table containing the extracted information from the driver's licenses.
<h2>Contributors</h2>

[<span>**Vaanry**</span>](https://github.com/Vaanry)
