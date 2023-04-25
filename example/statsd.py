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

import time
from pstatsd import Statsd

STATSD = None

class Statsds:

    def __init__(self, log_name = None):
        self.log_name = log_name
        Statsd.initialize(app_name="luca-model", host="status.pek01.rack.zhihu.com")
        self.statsd = Statsd

    def gauges(self, name, value, repeat=1):
        name = self.log_name + '.' + name
        try:
            self.statsd.gauge(name, value)
            for i in range(repeat - 1):
                time.sleep(0.005)
                self.statsd.gauge(name, value)
        except Exception as e:
            print(u'statsd.incr failed, detail msg: {}'.format(e))
        return None

    def incr(self, name, count=1):
        name = self.log_name + '.' + name
        try:
            self.statsd.incr(name, count = count)
        except Exception as e:
            print(u'statsd.incr failed, detail msg: {}'.format(e))
        return None

    def timing(self, name, value):
        name = self.log_name + '.' + name
        try:
            self.statsd.timing(name, value)
        except Exception as e:
            print(u'statsd.timing failed, detail msg: {}'.format(e))
        return None

def get_statsd(log_name = None):
    global STATSD
    if STATSD is not None:
        return STATSD
    STATSD = Statsds(log_name)
    return STATSD
