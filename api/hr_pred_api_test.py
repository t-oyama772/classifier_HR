import unittest
import requests
import json


class APITest(unittest.TestCase):
    URL = "http://localhost:5000/hr_pred_api/predict"
    DATA = {
        "satisfaction_level": 0.27,
        "last_evaluation": 0.61,
        "number_project": 3.0,
        "average_montly_hours": 213.0,
        "time_spend_company": 6.0,
        "salary_low": 1.0
    }

    def test_normal_input(self):
        # post requests
        response = requests.post(self.URL, json=self.DATA)
        # result
        print(response.text)  # Debug
        result = json.loads(response.text)  # JSON to dictionary
        # check status code : 200
        self.assertEqual(response.status_code, 200)
        # check status : OK
        self.assertEqual(result["status"], "OK")
        # check if the predicted value has a minus value
        self.assertTrue(0 <= result["predicted"])


if __name__ == "__main__":
    unittest.main()
