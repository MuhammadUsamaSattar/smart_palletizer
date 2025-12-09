# syntax=docker/dockerfile:1

FROM ros:kilted-ros-base AS builder
RUN apt-get update && \
    apt-get -y upgrade

WORKDIR /smart_palletizer

COPY src/ ./src/

RUN bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && \
             rosdep update && \
             rosdep install --from-paths src -yi && \
             colcon build"

FROM ros:kilted-ros-base AS runner
COPY --from=builder /smart_palletizer/install/ /smart_palletizer/install
WORKDIR /smart_palletizer

RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install pip && \
    apt-get -y install python3-venv

COPY src/ ./src/
RUN bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && \
             rosdep update && \
             rosdep install --from-paths src -yi && \
             rm -rf /var/lib/apt/lists/* src/"

ENV VENV_PATH=/opt/venv
RUN python3 -m venv $VENV_PATH
ENV PATH="$VENV_PATH/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir --no-input -r requirements.txt && \
    rm requirements.txt

RUN echo "source /opt/venv/bin/activate" >> ~/.bashrc && \
    echo "source /smart_palletizer/install/setup.bash" >> ~/.bashrc
