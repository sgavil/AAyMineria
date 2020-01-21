import numpy as np
from flask import Flask, render_template, request, jsonify
import mobile_price

app = Flask(__name__, static_url_path='')

num = 0

@app.route('/')
@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/main', methods=['GET', 'POST'])
def main():

    bateria_val = request.args.get('bateria_val', 0, type=float)
    blue_val = request.args.get('blue_val', 0, type=float)
    procesador_val = request.args.get('procesador_val', 0, type=float)
    sim_val = request.args.get('sim_val', 0, type=float)
    cam_fron_val = request.args.get('cam_fron_val', 0, type=float)
    cuatrog_val = request.args.get('cuatrog_val', 0, type=float)
    memoria_val = request.args.get('memoria_val', 0, type=float)
    profundidad_val = request.args.get('profundidad_val', 0, type=float)
    peso_val = request.args.get('peso_val', 0, type=float)
    nucleos_val = request.args.get('nucleos_val', 0, type=float)
    cam_prin_val = request.args.get('cam_prin_val', 0, type=float)
    res_alto_val = request.args.get('res_alto_val', 0, type=float)
    res_ancho_val = request.args.get('res_ancho_val', 0, type=float)
    ram_val = request.args.get('ram_val', 0, type=float)
    altura_val = request.args.get('altura_val', 0, type=float)
    anchura_val = request.args.get('anchura_val', 0, type=float)
    autonomia_val = request.args.get('autonomia_val', 0, type=float)
    tresg_val = request.args.get('tresg_val', 0, type=float)
    tactil_val = request.args.get('tactil_val', 0, type=float)
    wifi_val = request.args.get('wifi_val', 0, type=float)

    file_num = request.args.get('file_num', 0, type=int)

    a = np.array([[
        bateria_val,
        blue_val,
        procesador_val,
        sim_val,
        cam_fron_val,
        cuatrog_val,
        memoria_val,
        profundidad_val,
        peso_val,
        nucleos_val,
        cam_prin_val,
        res_alto_val,
        res_ancho_val,
        ram_val,
        altura_val,
        anchura_val,
        autonomia_val,
        tresg_val,
        tactil_val,
        wifi_val
        ]])
    np.savetxt('../ProcessedDataSet/user.csv', (a), delimiter=',')   

    res = mobile_price.calculate_range_price()

    f= open("static/result" + str(file_num) + ".txt", "w+")

    if res == 0:
        f.write("GAMA BAJA")
    
    elif res == 1:
        f.write("GAMA MEDIA")

    elif res == 2:
        f.write("GAMA ALTA")

    elif res == 3:
        f.write("GAMA SUPERIOR")
    
    f.close

    return jsonify(result=0)
    


if __name__ == "__main__":
    app.run(debug=True)