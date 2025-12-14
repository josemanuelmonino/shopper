import pandas as pd
import uuid
import random
from datetime import datetime

# --- 1. Generación de datos de posiciones de bolsas ---
def generate_bag_positions(num_points: int) -> pd.DataFrame:
    """Simula posiciones de bolsas para almacenar en la DB."""
    
    def get_coords(zone: str):
        if zone == "HOT_SPOT":
            return random.uniform(3.5, 6.5), random.uniform(5.5, 9.0)
        else:
            return random.uniform(0.0, 10.0), random.uniform(0.0, 10.0)
    
    records = []
    for i in range(num_points):
        zone = "HOT_SPOT" if random.random() < 0.4 else "TRANSIT"
        x, y = get_coords(zone)
        record = {
            "bag_id": f"BAG_{(i % 15 + 1):05d}",
            "customer_id": f"CUST_{(i % 40 + 1):05d}",
            "session_id": str(uuid.uuid4()),
            "x": x,
            "y": y,
            "timestamp": datetime.now().isoformat()
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    return df