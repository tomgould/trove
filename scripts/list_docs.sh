#!/bin/bash

# Define colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

TARGET_DIR="${1:-.}"

if [ ! -d "$TARGET_DIR" ]; then
    echo -e "${YELLOW}Error: Directory '$TARGET_DIR' not found.${NC}"
    exit 1
fi

echo -e "${BLUE}==================================================${NC}"
echo -e "${YELLOW}           DOCUMENT INVENTORY REPORT             ${NC}"
echo -e "${BLUE}==================================================${NC}"
echo -e "${GREEN}Target Folder:${NC} $(realpath "$TARGET_DIR")"
echo -e "${GREEN}Date:         ${NC} $(date)"
echo -e "${BLUE}==================================================${NC}"

# Variables to track directory changes
LAST_DIR=""

# Find documents, sort them by path so they group together
find "$TARGET_DIR" -type f \( -name "*.pdf" -o -name "*.doc" -o -name "*.docx" -o -name "*.txt" -o -name "*.xls" -o -name "*.xlsx" -o -name "*.ppt" -o -name "*.pptx" \) | sort | while read -r file; do

    # Get the directory of the current file
    CURRENT_DIR=$(dirname "$file")

    # If we have moved to a new directory, print a header
    if [ "$CURRENT_DIR" != "$LAST_DIR" ]; then
        echo -e "\n${BOLD}${BLUE}📁 FOLDER: ${CURRENT_DIR}${NC}"
        echo "--------------------------------------------------+-----------"
        printf "%-50s | %-10s\n" "FILE NAME" "SIZE"
        echo "--------------------------------------------------+-----------"
        LAST_DIR="$CURRENT_DIR"
    fi

    # Get file name and size
    file_name=$(basename "$file")
    size=$(du -h "$file" | cut -f1)

    # Truncate long filenames for the table
    display_name="$file_name"
    if [ ${#display_name} -gt 47 ]; then
        display_name="${display_name:0:44}..."
    fi

    printf "%-50s | %-10s\n" "$display_name" "$size"
done

echo -e "\n${BLUE}==================================================${NC}"
echo -e "${YELLOW}Scan Complete${NC}"
