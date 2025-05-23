from pydantic import BaseModel

class ApartmentFeatures(BaseModel):
    number_of_rooms: float
    district: int
    structure_type: int
    year_of_construction: float
    floor: float
    area: float
    quality: int
