import unittest
from analyse import analyse_function

class TestAnalyse(unittest.TestCase):
    def test_analyse_case_1(self):
        # cas simple déterministe : dernière rencontre gagnée par l'équipe à domicile
        input_data_1 = {
            'df': [
                {'Date': '2025-09-01', 'HomeTeam': 'TeamA', 'AwayTeam': 'TeamB', 'FTR': 'A', 'FTHG': 0, 'FTAG': 2},
                {'Date': '2025-09-10', 'HomeTeam': 'TeamC', 'AwayTeam': 'TeamD', 'FTR': 'H', 'FTHG': 3, 'FTAG': 1},
                {'Date': '2025-09-20', 'HomeTeam': 'HomeX', 'AwayTeam': 'AwayY', 'FTR': 'H', 'FTHG': 2, 'FTAG': 1}
            ],
            'force_simple': True
        }
        expected_output_1 = {
            'home': 'HomeX',
            'away': 'AwayY',
            'prob_be': [0.05, 0.10, 0.85],
            'recommended': 'Domicile'
        }
        self.assertEqual(analyse_function(input_data_1), expected_output_1)

    def test_analyse_case_2(self):
        # cas déterministe : dernière rencontre nul
        input_data_2 = {
            'df': [
                {'Date': '2025-08-01', 'HomeTeam': 'T1', 'AwayTeam': 'T2', 'FTR': 'H', 'FTHG': 1, 'FTAG': 0},
                {'Date': '2025-08-15', 'HomeTeam': 'T3', 'AwayTeam': 'T4', 'FTR': 'D', 'FTHG': 2, 'FTAG': 2}
            ],
            'force_simple': True
        }
        expected_output_2 = {
            'home': 'T3',
            'away': 'T4',
            'prob_be': [0.10, 0.80, 0.10],
            'recommended': 'Nul'
        }
        self.assertEqual(analyse_function(input_data_2), expected_output_2)

if __name__ == '__main__':
    unittest.main()