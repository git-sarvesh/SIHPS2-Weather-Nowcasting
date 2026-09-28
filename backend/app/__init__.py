"""SIHPS - Hyper-Local Weather Nowcasting System for India.

A unified, end-to-end, differentiable prototype that ingests multi-source
atmospheric data (INSAT-3D/3DR, IMDAA, IMD, SRTM/CartoDEM) and produces
terrain-aware, calibrated, explainable nowcasts of thunderstorm, cloudburst and
flash-flood risk at 0-6 h lead time.

Package layout
--------------
``app.ingestion``   satellite / reanalysis / terrain connectors + grid alignment
``app.models``      PyTorch CNN-ConvLSTM backbone, multi-task heads, uncertainty,
                    cross-hazard attention, and a NumPy reference backend
``app.services``    risk engine, explainability, calibration, alerts, inference
``app.api``         FastAPI routers (v1)
``app.tasks``       Celery ingestion / batch-inference / alert tasks
``app.db``          SQLAlchemy models + PostGIS DDL
"""

__version__ = "0.1.0"
