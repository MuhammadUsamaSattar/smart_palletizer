# syntax=docker/dockerfile:1

FROM ros:kilted-ros-base AS cacher
WORKDIR /smart_palletizer

COPY src ./src

RUN rosdep update --rosdistro $ROS_DISTRO && \
    cat <<EOF > /etc/apt/apt.conf.d/docker-clean && apt-get update
APT::Install-Recommends "0";
APT::Install-Suggests "0";
EOF

RUN bash -e <<'EOF'
declare -A types=(
  [exec]="--dependency-types=exec"
  [build]="")
for type in "${!types[@]}"; do
  rosdep install -y \
    --from-paths src \
    --reinstall \
    --simulate \
    ${types[$type]} \
    | grep 'apt-get install' \
    | awk '{print $4}' \
    | sort -u > /tmp/${type}_deps_list.txt
done
EOF


FROM ros:kilted-ros-base AS builder
WORKDIR /smart_palletizer
COPY --from=cacher /smart_palletizer/src ./src
COPY --from=cacher /tmp/build_deps_list.txt /tmp/build_deps_list.txt

RUN --mount=type=cache,target=/etc/apt/apt.conf.d,from=cacher,source=/etc/apt/apt.conf.d \
    --mount=type=cache,target=/var/lib/apt/lists,from=cacher,source=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    xargs apt-get install -y < /tmp/build_deps_list.txt
RUN bash -c "source /opt/ros/$ROS_DISTRO/setup.bash && \
    colcon build"


FROM ros:kilted-ros-core AS runner
WORKDIR /smart_palletizer
COPY --from=builder smart_palletizer/install ./install
COPY --from=cacher /tmp/exec_deps_list.txt /tmp/exec_deps_list.txt

RUN --mount=type=cache,target=/etc/apt/apt.conf.d,from=cacher,source=/etc/apt/apt.conf.d \
    --mount=type=cache,target=/var/lib/apt/lists,from=cacher,source=/var/lib/apt/lists \
    --mount=type=cache,target=/var/cache/apt,sharing=locked \
    xargs apt-get install -y < /tmp/exec_deps_list.txt
RUN echo "source /smart_palletizer/install/setup.bash" >> ~/.bashrc