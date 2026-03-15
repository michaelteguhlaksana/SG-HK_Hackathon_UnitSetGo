# Trading System

Algo trading system. The architecture is composed of:
- The Roostoo interface will act as the gateway to the Roostoo API. This will be:
    1. Retrieve price data
    2. Retrieve account and order data
    3. Sending and managing order 
    4. Portfolio sizing.
- The data-bus, probably something lightweight like SQLite (a bit conflicted with Redis too, but I think the memory permanence will be useful). Ideally something like Kafka, but it seems to be too heavy for an EC2 system. This will be the information holder and transmission line between the strategies and the broker interface. THis will:
    1. Store internal and/or important states (order status, balance, etc.)
    2. logs? Maybe? Keeping the "hot path" clear?
    3. Transmitting signals/orders and status to/between the strategies and the interface
    4. Transmitting data (price, external variables, etc.) to the strategies
- The strategies, probably more than 1, but likely simple. At most  a simple ML model or a NN. Will either create an order or a signal. Probably also just creating a signal and a separate "master" strategy will then manage the portfolio. This will:
    1. Create signal and/or order
    2. Ingest data from data bus