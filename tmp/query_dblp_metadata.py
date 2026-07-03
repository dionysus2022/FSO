import json
import ssl
import time
import urllib.parse
import urllib.request

import certifi


TITLES = [
    "Automatic Modulation Classification for OFDM Systems Using Bi-Stream and Attention-Based CNN-LSTM Model",
    "OFDM Modulation Classification Using Cross-SKNet With Blind IQ Imbalance and Carrier Frequency Offset Compensation",
    "Real-Time OFDM Signal Modulation Classification Based on Deep Learning and Software-Defined Radio",
    "Deep Learning-Based Automatic Modulation Classification With Blind OFDM Parameter Estimation",
    "Automatic Modulation Classification for Adaptive OFDM Systems Using Convolutional Neural Networks With Residual Learning",
    "Generative machine learning for robust free-space communication",
    "A Survey of Blind Modulation Classification Techniques for OFDM Signals",
    "Low Complexity OSNR Monitoring and Modulation Format Identification Based on Binarized Neural Networks",
    "CPAA Self-Supervised Cross-View Prediction With Automatic Augmentation for OFDM Modulation Classification",
]


for title in TITLES:
    query = urllib.parse.urlencode({"q": title, "format": "json", "h": 3})
    request = urllib.request.Request(
        f"https://dblp.org/search/publ/api?{query}",
        headers={"User-Agent": "paper-audit/1.0"},
    )
    print(f"\nQUERY {title}")
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(request, timeout=30, context=context) as response:
            payload = json.load(response)
        hits = payload.get("result", {}).get("hits", {}).get("hit", [])
        for hit in hits:
            print(json.dumps(hit.get("info", {}), ensure_ascii=False))
    except Exception as exc:
        print(f"ERROR {exc!r}")
    time.sleep(1.2)
