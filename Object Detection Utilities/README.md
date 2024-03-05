
# Data Preparation for YOLO Object Detection

This Python notebook provides a comprehensive set of functions for preprocessing data for object detection tasks, with a focus on preparing data for YOLO (You Only Look Once) models through several key steps.

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Steps](#steps)
   - [Step 0: Data Collection](#step-0-data-collection)
   - [Step 1: Data Specification](#step-1-data-specification)
   - [Step 2: Handling Missed Annotations](#step-2-handling-missed-annotations)
   - [Step 3: Label Processing](#step-3-label-processing)
   - [Step 4: Computing Aggregated Metrics](#step-4-computing-aggregated-metrics)
   - [Step 5: Assign Fold Numbers with Stratified K-Fold](#step-5-assign-fold-numbers-with-stratified-k-fold)
   - [Step 6: Split Data into Train, Test, and Validation Sets](#step-6-split-data-into-train-test-and-validation-sets)
   - [Step 7: Visualizing Class Distribution](#step-7-visualizing-class-distribution)
   - [Step 8: Organizing Data for YOLO](#step-8-organizing-data-for-yolo)
   - [Step 9: Creating YOLO Data Configuration File](#step-9-creating-yolo-data-configuration-file)
4. [Resources](#resources)
5. [Example Usage](#example-usage)
6. [Acknowledgements](#acknowledgements)

## Introduction

This notebook provides a set of functions to preprocess data for YOLO object detection models. The process involves collecting data from messy folders, specifying image and label paths, filtering images with missing annotations, and performing various preprocessing steps.

## Prerequisites

Make sure to install the required packages by running the following command:

```bash
pip install tqdm iterstrat ml-stratifiers pandas matplotlib numpy ruamel.yaml
```

## Steps

### Step 0: **Data Collection (`create_dataframe_and_folders` function):**

* Identifies image and label files within a folder.
* Creates a DataFrame with image paths and corresponding label paths.
* Handles missing labels or filename mismatches.

```python
# Collect the data from messy folders
df = create_dataframe_and_folders(input_folder_path, output_folder_path)
```

### Step 1: **Data Specification (`count_images`, `copy_files_to_folders` functions):**

* Counts images in a folder.
* Copies images and labels to designated folders based on the DataFrame within the metadata for each image.
* Offers options to append or overwrite existing files.

```python
# Build a DataFrame with metadata for each image
df = read_txt_from_df(df, task='det')
```

### Step 2: **Handling Missed Annotations (`get_missed_annot`, `save_missed_annot`, `drop_missed_annot` functions):**

* Creates an output folder for images with missed annotations (class label '-1').
* Filters and saves rows with missed annotations.
* Drops rows with missed annotations from the DataFrame.

```python
# Filter images with missed annotations
missed_annot_df = get_missed_annot(df, output_folder='missed_annot')
save_missed_annot(missed_annot_df, output_folder='missed_annot')
drop_missed_annot(df, missed_annot_df)
```

### Step 3: **Label Processing (`read_txt_from_df` function):**

* Reads information from text label files.
* Adds columns for bounding box coordinates (`x`, `y`, `w`, `h`) and class labels.
* Handles different label formats (detection vs. segmentation).

```python
# Preprocess DataFrame for model input
processed_df = process_dataframe(df)
aggregated_df = compute_aggregated_metrics(processed_df)
```

### Step 4: **Computing Aggregated Metrics (`compute_aggregated_metrics` function):**

* Calculates object count (`cnt`), average width (`avg_w`), average height (`avg_h`), and average aspect ratio (`avg_ratio`).

```python
# Compute aggregated metrics
aggregated_df = compute_aggregated_metrics(processed_df)
```

### Step 5: **Assign Fold Numbers with Stratified K-Fold (`assign_stratifiedKFold` function):**

* Implements Stratified K-Fold split for training, validation, and testing.

```python
# Assign fold numbers using Multilabel Stratified K-Fold
aggregated_df = assign_stratifiedKFold(aggregated_df)
```

### Step 6: **Split Data into Train, Test, and Validation Sets (`process_folds` function):**

* Concatenates folds and calculates data split percentages.

```python
# Split data into train, test, and validation sets
fold_range_train = range(8)
fold_val = 8
fold_test = 9
train_df, val_df, test_df = process_folds(aggregated_df, fold_range_train, fold_val, fold_test)
```

### Step 7: **Visualizing Class Distribution (`calculate_class_distribution`, `create_class_distributions_list`, `plot_class_distribution` functions):**

* Calculates class distribution for each split.
* Plots a stacked bar chart to visualize class distribution across splits.

```python
# Plot class distribution for each split
class_columns = aggregated_df.filter(like='class').columns
class_distributions_list = create_class_distributions_list(train_df, val_df, test_df, class_columns)
plot_class_distribution(class_distributions_list, class_names_dic)
```

### Step 8: **Organizing Data for YOLO (`create_output_folders`, `move_images` functions):**

* Creates output folders for images and labels based on the split type (train, val, test).
* Moves images and labels to their respective split folders.

```python
# Move images to respective folders based on YOLO folder structure
move_images(train_df, 'train')
move_images(test_df, 'test')
move_images(val_df, 'val')
```

### Step 9: **Creating YOLO Data Configuration File (`create_yaml_file` function):**

```python
# Create data.yaml file for dataset information
yaml_file_path = create_yaml_file(classes_list, output_folder_path='data_yolo_format/dataset/')
```

## Resources

- [Blog: Object Detection Stratification](https://jaidevd.com/posts/obj-detection-stratification/)
- [Paper: Improving Object Detection Performance](https://www.researchgate.net/publication/373451272_Improving_the_performance_of_object_detection_by_preserving_label_distribution)
- [GitHub Implementation](https://github.com/leeheewon-01/YOLOstratifiedKFold/tree/main)

## Example Usage

You can use the provided functions by following the example usage at the end of the notebook.

```python
main(task='det', mode='overwrite')
```

## Acknowledgements

Special thanks to the authors of the referenced blog, paper, and GitHub implementation for their valuable insights.

Feel free to adapt and modify the code according to your specific requirements. Happy preprocessing!

