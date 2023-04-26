# coding=utf-8
# Copyright 2020 The OpenBMB team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import time
from pstatsd import Statsd

STATSD = None

class Statsds:

    def __init__(self):

        node_name = os.getenv("NODE_NAME", "jeeves-hpc-gpu00").replace(".", "_")
        rank_name = (int)(os.getenv("RANK", "0")) // 8
        project_name = os.getenv("PROJECT_NAME", "no-project-name")
        job_id = os.getenv("JEEVES_JOB_ID", "000")
        status_prefix = "job-status.{}.{}.{}.{}.step".format(
            project_name,
            job_id,
            rank_name,
            node_name,
        )

        self.status_prefix = status_prefix
        Statsd.initialize(app_name="luca-model", host="status.pek01.rack.zhihu.com")
        self.statsd = Statsd

    def gauges(self, name, value, repeat=1):
        name = self.status_prefix + '.' + name
        try:
            self.statsd.gauge(name, value)
            for i in range(repeat - 1):
                time.sleep(0.005)
                self.statsd.gauge(name, value)
        except Exception as e:
            print(u'statsd.incr failed, detail msg: {}'.format(e))
        return None

    def incr(self, name, count=1):
        name = self.status_prefix + '.' + name
        try:
            self.statsd.incr(name, count = count)
        except Exception as e:
            print(u'statsd.incr failed, detail msg: {}'.format(e))
        return None

    def timing(self, name, value):
        name = self.status_prefix + '.' + name
        try:
            self.statsd.timing(name, value)
        except Exception as e:
            print(u'statsd.timing failed, detail msg: {}'.format(e))
        return None

def get_statsd():
    global STATSD
    if STATSD is not None:
        return STATSD

    STATSD = Statsds()
    return STATSD
