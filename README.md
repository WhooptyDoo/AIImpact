# GPT-4o Infrastructure Calculator

📊 **Quantify the infrastructure needed to serve GPT-4o at a planetary scale.**

This interactive web application allows you to explore the hardware, power, and cost implications of deploying GPT-4o to billions of users. Modify parameters and instantly see how GPU requirements, energy consumption, and financial costs scale.

---

## 🚀 Features

- 🔧 **Real-Time Calculator**  
  Adjust assumptions (users, tokens per query, model throughput, GPU specs, electricity cost) and instantly see infrastructure estimates.

- 📈 **Infrastructure Estimates**
  - GPUs Required (Average & Peak)
  - Power Draw (GW)
  - Daily to Annual Cost
  - Electricity Consumption (GWh/TWh)

- 🌍 **Global Comparison**
  - Compare GPT-4o’s annual power needs against countries like the US, China, India, Japan, etc.
  - See how many power plants are equivalent in scale.

- 📊 **Visual Analytics**
  - Cost over time (day, week, month, year)
  - Electricity consumption over time
  - Sensitivity analysis on cost drivers
  - Scalability plots with billions of users

- 📤 **Export Capabilities**
  - Download results and parameters in CSV format for external analysis.

---

## 📌 Sample Default Parameters

| Parameter                    | Default Value           |
|-----------------------------|--------------------------|
| Total Users                 | 8.0 billion              |
| Queries per User per Day    | 10                       |
| Tokens per Query            | 750                      |
| Throughput per H100 GPU     | 109 tokens/second        |
| H100 Power Consumption      | 700 W                    |
| Electricity Cost            | $0.15 / kWh              |

---

## 💡 Example Output (Under Default)

| Metric                      | Value                    |
|----------------------------|--------------------------|
| Daily Queries              | 80.0B                    |
| Tokens per Second          | ~694.4M                  |
| GPUs Required              | 6.37M (Avg), 12.74M (Peak)|
| Power Draw                 | 4.46 GW (Avg), 8.92 GW (Peak)|
| Daily Cost                 | $16.06M                  |
| Annual Cost                | $5.86B                   |
| Annual Energy Consumption  | 39.1 TWh                 |

---

## 🌐 Power Context

- Equivalent to **4 large nuclear plants**, **7 average coal plants**, or **11 gas plants**.
- Could power **3.67 million US homes** annually.
- Accounts for:
  - 🇺🇸 0.92% of US power
  - 🇨🇳 0.41% of China's
  - 🇮🇳 2.00% of India’s
  - 🇯🇵 3.86% of Japan’s

---

## 📊 Sensitivity & Scaling Insights

- Shows how costs scale with:
  - Increasing users
  - More queries per user
  - Higher token generation
- Analyze GPU and power needs at global scale (1B–10B users)

---

## 📝 Notes

- This is an **estimation tool** — real-world deployments require considerations such as:
  - Model optimization
  - Hardware utilization efficiency
  - Cooling, redundancy, and networking overhead
  - Variability in usage patterns

---

## 📂 Exports

- [x] Export parameter configurations as `.csv`
- [x] Export calculation results as `.csv`

---

## 🛠️ Built With

- HTML, CSS, JavaScript
- D3.js or Chart.js (for visualizations)
- Hosted on Replit or similar
**Website:** [yourwebsite.com]

