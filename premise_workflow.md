
The goal of this project is to create an interactive program which can run from start to finish, where "start" is considered "Ground Zero". "Ground Zero" is defined as having a folder composed of just pictures of the pages of a book. "Finish" is considered as the book fully processed into a readable PDF with bookmarks and also segmented into subcomponents with associated summaries.

The main stages of the program (currently) are as follows:

0. LLM Processor - **Main backbone of the program**
	- Loads in a prompt and documents to act on.
	- Loads in LLM to process documents given and responds according to the prompt.
	- Outputs a file containing the prompt response.

1. Document Compiler
	-  Receive a folder containing a series of images of a book.
	-  Crop pages to contain only the contents of the pages themselves.
	- Perform OCR on each image and compile the images into a PDF.

2. Document Processor
	- Receive a PDF.
	- Section off first 20-30 pages of PDF and possibly last 10-20 pages of PDF.
	- Use the LLM Processor to extract metadata from the document.
	- Adds metadata to the document information.
	- Adds bookmarks to the PDF according to the table of contents.

3. Document Slicer
	- Uses table of contents...

4. Summarizer
	- ...

---

| Stage | Input(s) | Outputs |
| --- | --- | --- |
| LLM Processor | list of filepaths (first is prompt, remaining are documents) | File containing response and filepath for said file |
| Document Compiler | Filepath for the folder containing book images | PDF document and filepath for said document |
| Document Processor | Filepath for the file to be processed | Metadata organized into a JSON file, complete with table of contents including page numbers |
| Document Slicer | Filepath
| Summarizer |



--- OLD ---
1. Receive a folder with images. These images are pictures of the open pages of a book against an unknown background. It is assumed the pages contrast sufficiently with the background. It is also assumed that the pages have some identifier in their name that indicates their order, and that no page is identified out of order.
2. Crop the images to contain only the face of the pages and excludes any backgrounds.
3. Splits the images so that there is only one page per image.
4. De-skews the image to align the page correctly.
5. Assembles the images into a PDF.
6. Performs OCR on the PDF.
7. Identifies the outline of the book. If there are sections of the book, it identifies those first. Then, within each section, it identifies the chapters within each section. It assembles this as a referenceable temp file.
8. The program splits the book (now a pdf) into each section. It outputs these as temporary files. It then splits each section into their individual chapters. If there are no sections, then it considers the entire book as a section and splits only on the chapters. These are outputted as their own pdfs. 
9. Goes through each chapter and, using LLMs, creates short and long summaries of each chapter. The idea and steps behind this process are already largely created and just need to be programmed.
10. Goes through each section (or book, if the entire book is considered a section) and creates a summary of the book as a whole. It also uses LLMs to create citations of the book.
11. Outputs the summaries as readable documents.

The resulting program is intended to be interactable by the user in a GUI window, such that the user can start the program at any stage and dictate which step to stop at. The program needs to create logs of each step of the process. The program needs to show when each step is performed and the actions being performed. The program needs to show how much time has passed and how long each step took to complete. Ideally, the program might also estimate how much time remaining.