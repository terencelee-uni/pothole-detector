

"""
TO DO:
Nearest Cross Street
https://stackoverflow.com/questions/6582479/finding-nearest-street-given-a-lat-long-location
https://csr.dot.ca.gov/index.php/Msrsubmit

Web bot for form https://www.youtube.com/watch?v=YbGAUEjTKg4


"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import requests

def getCross(lat, lon):
    GOOGLE_MAPS_API_URL = 'http://maps.googleapis.com/maps/api/geocode/json'

    params = dict(latlng='%s,%s' % (lat, lon), key=API_KEY)
        

    # Do the request and get the response data
    req = requests.get(GOOGLE_MAPS_API_URL, params=params)
    res = req.json()

    # Use the first result
    result = res['results'][0]['address_components'][1]['long_name']
    return result

    

from imgurpython import ImgurClient
client_id = 'YOUR CLIENT ID'
client_secret = 'YOUR CLIENT SECRET'
def submit(lat, lon, direction, dateTime, mask):
    browser = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    browser.get("https://csr.dot.ca.gov/index.php/Msrsubmit")

    desc = browser.find_element(By.ID, "typeDesc")
    desc.send_keys("Roadway - Pothole")
    desc.send_keys(Keys.ENTER)

    loc = browser.find_element(By.ID, "pac-input")
    locInput = "%d, %d" % (lat, lon)
    loc.send_keys(locInput)
    loc.send_keys(Keys.ENTER)

    direc = browser.find_element(By.ID, "dirTravel")
    direc.send_keys(direction)
    direc.send_keys(Keys.ENTER)

    crossStreet = getCross(lat, lon)
    cross = browser.find_element(By.ID, "crossStreet")
    cross.send_keys(crossStreet)
    cross.send_keys(Keys.ENTER)

    date = dateTime.split("|")[1]
    year = date.split("-")[0]
    month = date.split("-")[1]
    day = date.split("-")[2]
    date = "%d/%d/%d" % (month, day, year)
    dat = browser.find_element(By.ID, "situationNoticedDate")
    dat.send_keys(date)
    dat.send_keys(Keys.ENTER)

    time = dateTime.split("|")[0].split(":")[0]
    if time > 12:
        temp = "PM"
    else:
        temp = "AM"
    time = browser.find_element(By.ID, "situationNoticedTime")
    time.send_keys("%d %d" % (time, temp))
    time.send_keys(Keys.ENTER)

    nature = browser.find_element(By.ID, "situationDesc")
    client = ImgurClient(client_id, client_secret)
    image = client.upload_from_path(mask, anon=False)
    nature.send_keys("Pothole detected of significant size such that it poses a risk of vehicular damage and/or passenger health. See included link for photo: %s" % image)

    geog = browser.find_element(By.ID, "situationgGoeLoc")
    geog.send_keys("Pothole located at exact GPS coordinates provided.")

    email = browser.find_element(By.ID, "custEmail")
    email.send_keys("example@email.com")

    name = browser.find_element(By.ID, "custName")
    name.send_keys("Example Name Here")

    submit = browser.find_element(By.ID, "submitBttn")
    dat.send_keys(Keys.ENTER)

    browser.quit()


