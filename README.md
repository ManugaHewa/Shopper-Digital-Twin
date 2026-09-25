# Shopper Digital Twin

Predict what every customer would buy **before** a change happens.

Each shopper in a simulated Walmart-style general store gets a "twin", learned only from their browsing and purchase history. The twins are run forward in a simulator to answer what-if questions: *What happens to demand if this price goes up 10%? Where do shoppers go if this product is out of stock? Who buys the new product?*

Because the store and its shoppers are generated, every shopper's true tastes and price sensitivity are known. That makes it possible to check the twins' what-if forecasts against the real answer, something retailers can never do.

## How it works

- **Synthetic world:** ~5k products, ~100k shoppers, ~10M events, with a sealed answer key of true shopper parameters.
- **Twin model:** a per-shopper discrete choice model  
  `utility = taste match + habit + context - price sensitivity x price`  
  with a "buy nothing" option and hierarchical pooling for shoppers with little history.
- **Simulator:** day-by-day, vectorised over all twins, with FAISS candidate retrieval.
- **What-if engine:** re-simulate with the same random seeds and compare against the baseline.
- **Evaluation:** next-purchase accuracy, parameter recovery, what-if accuracy, calibration, speed.
- **Demo:** Streamlit app with a price slider and a chat to interview any twin.

## Docs

- [Project overview (PDF)](docs/Shopper_Digital_Twin_Overview.pdf)
- [Build plan and algorithm design](docs/plan.md)
- [Context log](docs/context.txt): detailed running record of decisions and implementation steps

## Status

Planning complete. Next: Milestone 1, the synthetic world generator.

Built in Python.
