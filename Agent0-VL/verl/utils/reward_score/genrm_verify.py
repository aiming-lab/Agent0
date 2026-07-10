import re

def find_answer_status(text):
    # Strictly match "The answer is correct/wrong."
    pattern = r"The answer is (correct|wrong)\.$"
    matches = []
    try:
        for match in re.finditer(pattern, text):
            phrase = match.group(0)
            start_pos = match.start()
            end_pos = match.end()
            matches.append((phrase, start_pos))
    except:
        matches = []
    return matches

    
def get_verification_score(solution_str: str, gt_judge: bool) -> float:
    status_list = find_answer_status(solution_str)
    if len(status_list) == 0 or len(status_list) > 1:
        return {"genrm_score": 0, "genrm_pred": "wrong"}
    else:
        status, _ = status_list[-1]
        if "correct" in status and gt_judge:
            return {"genrm_score": 1, "genrm_pred": "correct"}
        elif "correct" in status and not gt_judge:
            return {"genrm_score": 0, "genrm_pred": "correct"}
        elif "wrong" in status and not gt_judge:
            return {"genrm_score": 1, "genrm_pred": "wrong"} 
        else:
            return {"genrm_score": 0, "genrm_pred": "wrong"}