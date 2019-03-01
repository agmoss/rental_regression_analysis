import pandas as pd

class Query:
    """Get data from db"""

    def __init__(self,conn):
        # Database connection 
        self.conn = conn

    def data_for_analysis(self):
        # Read in the data from db
        try:

            # Read in a dataframe from the rentfaster database
            df = pd.read_sql('SELECT price,type,sq_feet,community,quadrant,bedrooms,den,baths,cats,dogs,utilities_included FROM rental_data', con=self.conn)

            # Not of use for analysis: intro, link, marker, phone, phone_2, preferred_contact, slide, thumb, thumb_2, title, website

            # Variables
            # discrete categorical attributes: address, availability, avdate, dity, community, den, location, province, quadrant, rented, status, type, utilities_included
            # discrete numeric attributes: address_hidden,baths,bedroons,cats,dogs, email, id, ref_id, userId
            # continuous attributes: latitude, longitude, price, sq_feet, 

            # price per foot column
            df['price_per_foot'] = df['price'] / df['sq_feet']

            return df

        except Exception:
            raise
