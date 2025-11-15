import gdown
import os

urls = [
    "https://drive.google.com/file/d/1oJ0tTpGtcrkUO3CThFuvBbPlJiZpSwin/view?usp=drive_link",
    "https://drive.google.com/file/d/1DznhFAwQbEBfSM-2Aq1sjvgVW5GEfViN/view?usp=drive_link",
    "https://drive.google.com/file/d/1vB1cOZlO4MQ7c503-UU109mMbClsf5Lq/view?usp=drive_link"
]

outputs = [
    "../data/fan_varying_rpm.dat",
    "../data/fan_varying_rpm_turning.dat",
    "../data/fan_const_rpm.dat"
]

for url, output in zip(urls, outputs):
    if not os.path.exists(output):
        gdown.download(url=url, output=output, fuzzy=True)
    elif os.path.exists(output):
        print("Already exists, skipping.")