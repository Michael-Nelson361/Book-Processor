
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
	- Uses table of contents to create extracts of the document according to "Part", "Chapter", and "Section"
	- "Part" is considered to have an entire page dedicated to the heading
	- "Chapter" is considered to have a heading with content following the heading on the same page
	- "Section" is considered to be a formatted heading surrounded by content

4. Summarizer
	- Given a document...:
	- Extract chapter context (single short paragraph giving explanation of the upcoming content)
	- Analyze document to build outline of information
	- Analyze document to extract three different perspectives presented (if similar perspectives, throw in randomness of foreign or contradicting perspectives)
	- Generate summaries according to the outline, as understood by each perspective
	- Merge summaries of each perspectives according to similarities
	- Generate a single sentence representative of each summary paragraph
	- Create a single summary paragraph summarizing the summary paragraphs
	- Create a shortened single sentence summary of the summary paragraph
5. UI Wrapper
	- Enables all stages to be run
	- Extends functionality of the program to enable extra features

---

| Stage | Input(s) | Outputs |
| --- | --- | --- |
| LLM Processor | list of filepaths (first is prompt, remaining are documents) | File containing response and filepath for said file |
| Document Compiler | Filepath for the folder containing book images | PDF document and filepath for said document |
| Document Processor | Filepath for the file to be processed | Metadata organized into a JSON file, complete with table of contents including page numbers |
| Document Slicer | Filepath of the file to be split | Folder(s) containing the split files and filepath for master directory |
| Summarizer | ... | ... |

---
Ideas to implement:
- Allow for an override of functionality
	- E.g., user wants to find certain information like where are polar coordinates covered in a calculus book or does a book on anthropology cover evolution?
	- Or can a user get only contexts or short summaries of the chapters?

---

The resulting program is intended to be interactable by the user in a GUI window, such that the user can start the program at any stage and dictate which step to stop at. The program needs to create logs of each step of the process. The program needs to show when each step is performed and the actions being performed. The program needs to show how much time has passed and how long each step took to complete. Ideally, the program might also estimate how much time remaining.