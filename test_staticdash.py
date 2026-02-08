#!/usr/bin/env python3
import staticdash
import plotly.graph_objects as go

# Create a simple plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='lines'))
fig.update_layout(title="Test Plot")

# Create report
report = staticdash.Report("Test Dashboard")
report.heading("Welcome", level=1)
report.text("This is a test of staticdash")
report.plot(fig)

# Save
report.save("test_report.html")
print("Saved to test_report.html")
