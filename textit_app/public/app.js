/**
 * Created by Noga on 18/06/2017.
 */

let app = angular.module('myApp', ['ngRoute', 'LocalStorageModule']);
//-------------------------------------------------------------------------------------------------------------------

app.config(function (localStorageServiceProvider) {
    localStorageServiceProvider.setPrefix('node_angular_App');
});

//-------------------------------------------------------------------------------------------------------------------
app.config(['$locationProvider', function($locationProvider) {
    $locationProvider.hashPrefix('');
}]);

//-------------------------------------------------------------------------------------------------------------------

app.config( ['$routeProvider', function($routeProvider) {
    $routeProvider
        .when("/", {
            templateUrl : "views/home.html",
            controller : "homeController"
        })
        .when("/lists", {
            templateUrl : "views/lists.html",
            controller: 'listsController',
        })
        .when("/about", {
            templateUrl : "views/about.html",
        })
        .otherwise({redirect: '/',
        });
}]);
