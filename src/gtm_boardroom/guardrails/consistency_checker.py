import re
import numpy as np
from typing import Dict, Any, List

class ConsistencyChecker:
    """
    A utility class designed to validate numerical values found in AI agent responses
    against a ground truth pool derived from SHAP information, market context, and
    optimization results. It handles various data types including NumPy scalars and arrays.
    """

    @staticmethod
    def _flatten_data(data: Any) -> List[float]:
        """
        Recursively extracts all numerical values from various data structures (lists, tuples,
        NumPy arrays, dictionaries) and converts them to floats. Non-finite values (NaN, Inf)
        are filtered out.
        Args:
            data (Any): The input data structure to flatten.
        Returns:
            List[float]: A list containing all finite numerical values found in the input data.
        """
        nums = []
        
        # 1. Handle NumPy arrays or Python lists
        if isinstance(data, (list, tuple, np.ndarray)):
            for item in data:
                nums.extend(ConsistencyChecker._flatten_data(item))
        
        # 2. Handle Dictionaries
        elif isinstance(data, dict):
            for v in data.values():
                nums.extend(ConsistencyChecker._flatten_data(v))
        
        # 3. Handle Numbers (Python int/float AND NumPy int/float)
        elif isinstance(data, (int, float, np.number)):
            # Convert to standard float, filter out NaNs or Infs
            val = float(data)
            if np.isfinite(val):
                nums.append(val)
        
        # 4. Handle Booleans (Often used in market_context)
        elif isinstance(data, bool):
            nums.append(float(data))

        return nums

    @staticmethod
    def validate_response(text: str, shap_info: Dict, market_context: Dict, opt_results: Dict) -> Dict[str, Any]:
        """
        Validates the Agent's natural language response by checking if numerical values
        mentioned in the text are present within a unified ground truth pool.
        The ground truth pool is constructed from SHAP information, market context, and
        optimization results. It accounts for minor rounding differences and common scaling
        issues (e.g., percentages, k-notation such as "176k").
        Args:
            text (str): The natural language response text from the AI agent.
            shap_info (Dict): Dictionary containing SHAP values or related model diagnostics.
            market_context (Dict): Dictionary describing the current market conditions and strategic context.
            opt_results (Dict): Dictionary containing the results of an optimization process.
        Returns:
            Dict[str, Any]: A dictionary indicating whether the response is valid (`is_valid`),
                            a list of detected hallucinated values (`hallucinated_values`), and
                            an error message if hallucinations are found (`error_msg`).
        """
        # Build the Ground Truth Pool from all sources
        raw_pool = []
        for source in [shap_info, market_context, opt_results]:
            raw_pool.extend(ConsistencyChecker._flatten_data(source))
        
        # Get absolute values and lose the signs
        ground_truth_pool = [abs(v) for v in raw_pool]

        # Context-gated k-notation: a number written as "176k" / "176 thousand" cites
        # value*1000. Only suffixed numbers earn the x1000 match, and only within one
        # unit of their last written digit ("176k" -> 176,000 +/- 1,000; "176.4k" ->
        # +/- 100), which accepts round-to-nearest and truncation of a real pool value
        # but keeps a bare "176" from ever silently matching 176,432.
        k_cited = {}
        for m in re.finditer(r"(?<![\d.\w])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)\s*(?:[kK]\b|[tT]housand\b)", text):
            digits = m.group(1).replace(',', '')
            decimals = len(digits.split('.')[1]) if '.' in digits else 0
            k_cited[abs(float(digits))] = 1000.0 * 10.0 ** (-decimals)

        # Clean and extract numbers from LLM text, format to remove commas and currency symbols first
        clean_text = text.replace(',', '').replace('$', '').replace('%', '')
        found_in_text = re.findall(r"[-+]?\d*\.\d+|\d+", clean_text)
        
        # Convert to float and absolute value
        text_nums = []
        for n in found_in_text:
            try:
                text_nums.append(abs(float(n)))
            except ValueError:
                continue

        hallucinations = []
        # Tolerance: 1.1 allows for rounding (e.g., 799.8 in data vs 800 in text)
        tolerance = 1.1 

        for num in text_nums:
            # Skip indices and common low-value markdown formatting numbers, these are found in bullet points for LLMs
            if num in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]:
                continue
            
            match_found = False
            for gt in ground_truth_pool:
                # Direct/Rounded match
                if abs(num - gt) < tolerance:
                    match_found = True; break
                
                # Percentage scaling (LLM says 5 for data 0.05)
                if abs(num - (gt * 100)) < tolerance:
                    match_found = True; break
                
                # Inverse Scaling (LLM says 0.12 for data 12.0)
                if abs(num - (gt / 100)) < tolerance:
                    match_found = True; break

                # k-notation (LLM says "176k" for data 176432.11) - only for
                # numbers the text actually wrote with a k/thousand suffix
                if num in k_cited and abs(num * 1000 - gt) < k_cited[num]:
                    match_found = True; break

            if not match_found:
                hallucinations.append(num)

        return {
            "is_valid": len(hallucinations) == 0,
            "hallucinated_values": hallucinations,
            "error_msg": f"Detected numbers not in Ground Truth: {hallucinations}" if hallucinations else None
        }