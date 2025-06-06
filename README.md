# Book-Processor

## Project Description:

This project is mainly to aid me in my own studies. This program will take a batch of user images, process them, and turn them into a full-fledge PDF with section and chapter bookmarks.

---

## Goals:
- 

## Program Workflow
1. Collect pictures of user images (ideally all in one folder)
2. Preprocess images (crop/deskew/split)
3. Compile images into PDF
4. Perform OCR on images
5. Post-process PDF     
    a. Verify pages are organized properly    
    b. Add bookmarks    
    c. Use internal document information to add metadata

### Optional Additional Workflows:
6. Segment sections and chapters into their own PDFs.
7. Cycle through chapters and create summaries of them.     
    a. Generate an introductory context of the chapter
    b. Generate an outline of the chapter   
    c. Provide the outline and the document to two different LLMs to generate summaries of the chapter  
    d. Provide the summaries to a third LLM to merge the summaries  
    e. 
8. Output 

## To-Do:
- [ ] Build program section to import images.
    - Ideally the user just enters or imports a folder where they've put all the images.
    - [ ] Add a warning note to tell the user to make sure only the images for the book are in the folder.
- [ ] Build the program to...

## Acknowledgements:
- 