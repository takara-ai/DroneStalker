import os

def parse(base_path: str):
    entries = []
    coords_file = os.path.join(base_path, "coordinates.txt")

    with open(coords_file, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            time_str, rest = line.split(":", 1)
            name = time_str.replace('.', '')

            if name[0] == "0":
                name = name[1:]

            parts = [p.strip() for p in rest.split(",")]

            x1 = float(parts[0])
            y1 = float(parts[1])
            x2 = float(parts[2])
            y2 = float(parts[3])
            label = int(parts[4])
            drone_name = parts[5]

            entries.append({
                "name": name,
                "x1": x1,
                "x2": x2,
                "y1": y1,
                "y2": y2,
                "label": label,
                "drone_name": drone_name
            })

    return entries


def path(seq, frame):
    return f"Video_{seq}_frame_{frame}.png"