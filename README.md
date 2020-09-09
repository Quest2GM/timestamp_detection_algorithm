# Timestamp Detection and Recognition Algorithm
Check out the research paper in the repository for a complete breakdown of the algorithm!

## The Algorithm
![](outputs/process.png)

## Digit Detection
![](outputs/time_stamp_animation.gif)

### YOLOv3 Timestamp Localization Performance
| Detection          | Images        |
|:------------------:|:-------------:|
| Correct Detection  | 754           |
| False Detection    | 5             |
| No Detection       | 2             |
| Partial Detection  | 17            |
| **Accuracy**       | **96.9%**     |

### Digit Recognition Algorithm Performance
| Detection               | Images        |
|:-----------------------:|:-------------:|
| Correct                 | 774           |
| Acceptable              | 177           |
| Incorrect               | 58            |
| **Accuracy**            | **94.1%**     |

(Note: "Acceptable" is defined as cases where the detection is off by one or two digits, but the timestamp's date can still be understood by a human. These cases arise in part due to partial detection in YOLOv3 localization.)

### Speed
Averages around 3 seconds for localization and detection combined.

## Acknowledgements
Thank you to the Systems Analysis and Forecasting Office team at the Ministry of Transportation for the support throughout the project. Another thank you to [PyLessons](https://pylessons.com/) for the amazing YOLOv3 program and the tutorials.