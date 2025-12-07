# syntax=docker/dockerfile:1
ARG FROM_IMAGE=osrf/ros:kilted-desktop

FROM ${FROM_IMAGE}
RUN sudo apt-get update && \
sudo apt-get -y upgrade

RUN sudo apt-get -y install pip && \
sudo apt-get -y install python3-venv

WORKDIR /smart_palletizer
COPY src/ ./src/
RUN rosdep update && \
rosdep install --from-paths src -yi
RUN bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && colcon build" && \
    rm -rf /var/lib/apt/lists/* src build log

ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --no-input -r requirements.txt && \
    rm requirements.txt

RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc && \
    echo "source /smart_palletizer/install/setup.bash" >> ~/.bashrc
