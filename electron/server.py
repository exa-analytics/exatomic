import json
import uuid
import functools
import datetime as dt
import exatomic
import tornado.ioloop
import tornado.web
from tornado.websocket import WebSocketHandler

@functools.lru_cache(maxsize=1)
def get_unis():
    from exatomic import gaussian
    
    trj_file = 'H2O.traj.xyz'
    orb_file = 'g09-ch3nh2-631g.out'
    nmr_file = 'g16-nitromalonamide-6-31++g-nmr.out'
    
#    trj = exatomic.XYZ(
#        exatomic.base.resource(trj_file)).to_universe()
    
    orb = gaussian.Output(
        exatomic.base.resource(orb_file)).to_universe()
    orb.add_molecular_orbitals()
    
#    nmr = gaussian.Output(
#        exatomic.base.resource(nmr_file)).to_universe()
#    nmr.tensor = nmr.nmr_shielding

    return orb # trj, orb, nmr


class MainHandler(tornado.web.RequestHandler):

    def set_default_headers(self):
        self.set_header('Access-Control-Allow-Origin', '*')
        self.set_header('Access-Control-Allow-Headers', 'X-requested-with')
        self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.set_header('content-type', 'application/json')

    def get(self):
        # trj, orb, nmr = get_unis()
        orb = get_unis()
        traits = exatomic.widgets.traits.atom_traits(orb.atom)
        self.write(json.dumps(traits))
        print("dumped atom", dt.datetime.now())

    def options(self):
        self.set_status(204)
        self.finish()



class WSHandler(WebSocketHandler):
    _live = {'count': 0}

    @property
    def log(self):
        import logging
        logging.basicConfig()
        return logging.getLogger('ws')

    def open(self):
        """Keep track of all live connections"""
        self.id = str(uuid.uuid4())
        self._live['count'] += 1
        self._live[self.id] = {'id': self.id, 'count': 0}
        msg = f"opened: {self.id}"
        self.log.info(msg)
        self.write_message(json.dumps({'id': self.id}))

    def on_close(self):
        self.log.info(f"closed: {self.id}")
        self._live.pop(self.id)

    def check_origin(self, origin):
        self.log.warning(f"ALLOWING ALL CORS: {origin}")
        return True

    async def _incref(self):
        """Keep count of all requests from client"""
        self._live[self.id]['count'] += 1


    async def on_message(self, msg):
        """Route incoming websocket message to
        appropriate handler.
        """
        try:
            # self.log.info(f"receiving {msg}")
            msg = json.loads(msg)
            self.log.info(f"received incoming msg={msg}")
        except Exception as e:
            self.log.error(f"error decoding={e}")
        await self._incref()
        self.log.info(self._live[self.id])


def make_app():
    return tornado.web.Application([
        (r'/get', MainHandler),
        (r'/ws', WSHandler)
    ])

if __name__ == '__main__':
    get_unis()
    app = make_app()
    app.listen(8888)
    print("app is live")
    tornado.ioloop.IOLoop.current().start()
